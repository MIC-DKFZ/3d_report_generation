"""Evaluation cadence and training termination callbacks."""

import math
from pathlib import Path

from transformers import TrainerCallback

from vlm.utils import logger


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
    def _disable_eval_and_save(control):
        control.should_evaluate = False
        control.should_save = False
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._should_run_on_epoch(args, state):
            return self._enable_eval_and_save(control)
        return self._disable_eval_and_save(control)
