"""Evaluation cadence and training termination callbacks."""

import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import distributed as dist
from transformers import TrainerCallback

from vlm.utils import logger
from vlm.utils.ddp import is_main_process

# Creating this file in the run directory asks the running job to save and stop.
STOP_FILENAME = "STOP_TRAINING"
# Written when an evaluation is skipped before the time limit; the next job runs it after resuming.
PENDING_EVALUATION_FILENAME = "PENDING_EVALUATION"


class TerminateOnNaNLossCallback(TrainerCallback):
    def __init__(self, threshold: int = 3, filename: str = "TRAINING_FAILED.txt"):
        self.threshold = threshold
        self.filename = filename
        self.bad_value_counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        current_loss = logs.get("loss")
        if current_loss is None:
            return

        is_bad_value = math.isnan(current_loss) or math.isinf(current_loss) or current_loss == 0.0

        if is_bad_value:
            self.bad_value_counter += 1
        else:
            self.bad_value_counter = 0

        if self.bad_value_counter >= self.threshold:
            file_path = Path(args.output_dir, self.filename)
            try:
                with open(file_path, "w") as f:
                    f.write("Status: ABORTED\n")
                    f.write(
                        f"Reason: Loss instability detected for {self.threshold} consecutive logs.\n"
                    )
                    f.write(f"Final Value: {current_loss}\n")
                    f.write(f"Global Step: {state.global_step}\n")

                logger.error_main(f"\nABORTING: Crash report written to: {file_path}")
            except Exception as e:
                logger.error_main(f"ABORTING: Could not write crash report: {e}")

            raise RuntimeError(
                f"Training aborted! Loss was {current_loss} for {self.threshold} consecutive logs. "
                f"See {file_path} for details."
            )


class PhaseEndCallback(TrainerCallback):
    """Stop training and save a checkpoint after the last epoch of a training phase."""

    def __init__(self, end_epoch: int):
        self.end_epoch = end_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if round(state.epoch) >= self.end_epoch:
            control.should_training_stop = True
            control.should_save = True
        return control


class EveryNEpochsCallback(TrainerCallback):
    """
    Run evaluation every ``n`` epochs, using ``eval.eval_every_n_epochs`` as ``n``.
    This enables an epoch-based evaluation cadence other than every epoch.

    Evaluated epochs are always saved, so the best checkpoint has its metrics. Other epochs
    are saved as 'save_strategy' says, which keeps resumes from losing up to 'n' epochs.
    """

    def __init__(self, n: int = 1):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("The evaluation epoch interval must be a positive integer.")
        self.n = n

    def _get_epoch(self, state):
        return int(round(state.epoch)) if state.epoch is not None else None

    def _is_final_epoch(self, args, epoch: int | None) -> bool:
        return epoch is not None and epoch >= int(round(args.num_train_epochs))

    def _should_run_on_epoch(self, args, state) -> bool:
        epoch = self._get_epoch(state)
        if self._is_final_epoch(args, epoch):
            return True
        return epoch is not None and epoch % self.n == 0

    @staticmethod
    def _enable_eval_and_save(control):
        control.should_evaluate = True
        control.should_save = True
        return control

    @staticmethod
    def _disable_eval(control):
        control.should_evaluate = False
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._should_run_on_epoch(args, state):
            return self._enable_eval_and_save(control)
        return self._disable_eval(control)


def slurm_job_end_time(attempts: int = 5, retry_seconds: float = 30) -> float | None:
    """
    Return the Slurm end time of the current job as a Unix timestamp, or None outside Slurm.

    Slurm 23.02 and later set it as 'SLURM_JOB_END_TIME'; otherwise 'scontrol' is asked, with
    retries, because Slurm commands time out when the controller is busy.
    """
    job_id = os.getenv("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        return float(os.environ["SLURM_JOB_END_TIME"])
    except (KeyError, ValueError):
        pass
    for attempt in range(attempts):
        if attempt:
            time.sleep(retry_seconds)
        try:
            output = subprocess.run(
                ["scontrol", "show", "job", job_id], capture_output=True, text=True, timeout=60
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for field in output.split():
            if field.startswith("EndTime="):
                try:
                    return datetime.fromisoformat(field.removeprefix("EndTime=")).timestamp()
                except ValueError:
                    return None
    logger.warning_main(f"Could not read the end time of Slurm job {job_id}; no stop before it.")
    return None


class GracefulStopCallback(TrainerCallback):
    """
    Save a checkpoint and stop when the run directory contains 'STOP_TRAINING' or the Slurm
    time limit comes close, so a follow-up job continues without losing training progress.

    Rank 0 decides every 'check_every_n_steps' steps and at every epoch end, and all ranks
    follow its decision. An evaluation is skipped instead of started when it might not finish
    before the time limit; 'PENDING_EVALUATION' in the run directory then records its step, and
    the job that resumes runs and saves it after its first step. Add this callback after
    'EveryNEpochsCallback', so it sees which epochs evaluate.
    """

    def __init__(
        self,
        stop_file: str | Path,
        end_time: float | None = None,
        margin_seconds: float = 20 * 60,
        evaluation_budget_seconds: float = 20 * 60,
        check_every_n_steps: int = 10,
    ):
        self.stop_file = Path(stop_file)
        self.pending_file = self.stop_file.with_name(PENDING_EVALUATION_FILENAME)
        self.end_time = end_time
        self.margin_seconds = margin_seconds
        self.evaluation_budget_seconds = evaluation_budget_seconds
        self.check_every_n_steps = check_every_n_steps
        self.stopped = False
        self._evaluation_start = None
        self._pending_checked = False
        self._running_pending_evaluation = False

    @classmethod
    def from_environment(cls, output_dir: str | Path) -> "GracefulStopCallback":
        """Use the Slurm end time and the 'VLM_STOP_*' minute settings of the environment."""
        end_time = slurm_job_end_time() if is_main_process() else None
        if end_time is not None:
            logger.info_main(
                f"The Slurm job ends at {datetime.fromtimestamp(end_time):%Y-%m-%d %H:%M}; "
                "training saves and stops before it."
            )
        return cls(
            Path(output_dir, STOP_FILENAME),
            end_time=end_time,
            margin_seconds=60 * float(os.getenv("VLM_STOP_MARGIN_MINUTES", "20")),
            evaluation_budget_seconds=60 * float(os.getenv("VLM_STOP_EVALUATION_MINUTES", "20")),
        )

    def _seconds_left(self) -> float | None:
        return None if self.end_time is None else self.end_time - time.time()

    def _stop_reason(self, seconds_needed: float) -> str | None:
        if self.stop_file.exists():
            return f"'{self.stop_file}' exists"
        seconds_left = self._seconds_left()
        if seconds_left is not None and seconds_left < seconds_needed:
            return f"the Slurm time limit is {seconds_left / 60:.0f} minutes away"
        return None

    @staticmethod
    def _agree(args, decision: bool) -> bool:
        """Broadcast rank 0's decision, so all ranks stop at the same step."""
        if not (dist.is_available() and dist.is_initialized()):
            return decision
        device = args.device if dist.get_backend() == "nccl" else torch.device("cpu")
        flag = torch.tensor([int(decision)], device=device)
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def _stop(self, args, state, control, reason: str | None):
        logger.warning_main(
            f"Saving and stopping at step {state.global_step}: {reason or 'requested by rank 0'}."
        )
        self.stopped = True
        control.should_save = True
        control.should_training_stop = True
        return control

    def _pending_step(self) -> int | None:
        try:
            return int(self.pending_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None

    def _run_pending_evaluation(self, args, state, control):
        """After the first step of a resumed job, run the evaluation the previous job skipped."""
        self._pending_checked = True
        pending_step = self._pending_step() if is_main_process() else None
        if pending_step is not None and pending_step >= state.global_step:
            # Resumed from before the skipped evaluation: its epoch end evaluates as usual.
            self.pending_file.unlink(missing_ok=True)
            pending_step = None
        if not self._agree(args, pending_step is not None):
            return
        if is_main_process():
            logger.info_main(
                f"Running the evaluation skipped at step {pending_step} before the time limit."
            )
        self._running_pending_evaluation = True
        self._evaluation_start = time.time()
        control.should_evaluate = True
        # Save the evaluated weights, so the checkpoint selection can choose them.
        control.should_save = True

    def on_step_end(self, args, state, control, **kwargs):
        if not self._pending_checked:
            self._run_pending_evaluation(args, state, control)
        if self.stopped or state.global_step % self.check_every_n_steps:
            return control
        reason = self._stop_reason(self.margin_seconds) if is_main_process() else None
        if self._agree(args, reason is not None):
            return self._stop(args, state, control, reason)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        # A stop inside an epoch also ends the epoch loop; its evaluation follows the resume.
        if self.stopped and not math.isclose(state.epoch, round(state.epoch), abs_tol=1e-6):
            control.should_evaluate = False
            return control
        # A requested stop still evaluates the epoch first, unless the time limit is too close.
        if control.should_evaluate:
            seconds_left = self._seconds_left() if is_main_process() else None
            needed = self.margin_seconds + self.evaluation_budget_seconds
            if self._agree(args, seconds_left is not None and seconds_left < needed):
                logger.warning_main(
                    f"Skipping the evaluation at epoch {round(state.epoch)}: the Slurm time "
                    "limit is too close."
                )
                control.should_evaluate = False
                if is_main_process():
                    self.pending_file.write_text(f"{state.global_step}\n")
                if not self.stopped:
                    self._stop(args, state, control, "no time left to evaluate")
                return control
            self._evaluation_start = time.time()
        if not self.stopped:
            reason = self._stop_reason(self.margin_seconds) if is_main_process() else None
            if self._agree(args, reason is not None):
                self._stop(args, state, control, reason)
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        if self._running_pending_evaluation:
            self._running_pending_evaluation = False
            if is_main_process():
                self.pending_file.unlink(missing_ok=True)
        # Budget the next evaluation by the longest one so far.
        if self._evaluation_start is not None:
            duration = time.time() - self._evaluation_start
            self.evaluation_budget_seconds = max(self.evaluation_budget_seconds, duration)
            self._evaluation_start = None
        return control
