"""A stop request or a close Slurm time limit saves and stops, and a follow-up job continues."""

import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from accelerate.state import AcceleratorState
from test_training_phases import MAX_STEPS, flat_parameters, train, uninterrupted  # noqa: F401
from transformers import TrainerCallback, TrainerControl

from vlm.training.callbacks import PENDING_EVALUATION_FILENAME, STOP_FILENAME, GracefulStopCallback
from vlm.training.callbacks import control as control_module


class RequestStopAtStep(TrainerCallback):
    """Create the stop file at a step, as a follow-up job or a person would."""

    def __init__(self, step, stop_file):
        self.step = step
        self.stop_file = stop_file

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            Path(self.stop_file).touch()


def stop_by_file(output_dir, step, **kwargs):
    stop_file = Path(output_dir, STOP_FILENAME)
    stop = GracefulStopCallback(stop_file, check_every_n_steps=1)
    _, trained, steps = train(
        output_dir,
        callbacks=[RequestStopAtStep(step, stop_file), stop],
        stop_requested=lambda: stop.stopped,
        **kwargs,
    )
    return stop, trained, steps


def test_stop_file_saves_and_resuming_matches_uninterrupted_training(tmp_path, uninterrupted):
    _, _, expected_parameters, expected_steps = uninterrupted
    stopped_dir = tmp_path / "stopped"
    stop, trained, steps = stop_by_file(stopped_dir, step=6, save_strategy="no")

    assert stop.stopped and trained
    assert sorted(steps) == list(range(6))
    # Without the stop, only the phase end would have been saved.
    assert sorted(path.name for path in stopped_dir.glob("checkpoint-*")) == [
        "checkpoint-4",
        "checkpoint-6",
    ]

    model, trained, steps = train(tmp_path / "resumed", stopped_dir / "checkpoint-6")
    assert trained
    assert steps == {step: value for step, value in expected_steps.items() if step >= 6}
    torch.testing.assert_close(flat_parameters(model), expected_parameters)


def test_stop_at_the_last_step_of_a_phase_does_not_start_the_next_phase(tmp_path):
    stop, _, steps = stop_by_file(tmp_path, step=4)
    assert stop.stopped
    assert sorted(steps) == list(range(4))


def test_close_time_limit_stops_at_the_next_check(tmp_path):
    stop = GracefulStopCallback(
        tmp_path / STOP_FILENAME, end_time=time.time() + 30, margin_seconds=60,
        check_every_n_steps=3,
    )
    _, _, steps = train(tmp_path, callbacks=[stop], stop_requested=lambda: stop.stopped)
    assert stop.stopped
    assert sorted(steps) == [0, 1, 2]


def test_without_stop_file_or_time_limit_training_runs_to_the_end(tmp_path):
    stop = GracefulStopCallback(tmp_path / STOP_FILENAME, check_every_n_steps=1)
    _, _, steps = train(tmp_path, callbacks=[stop], stop_requested=lambda: stop.stopped)
    assert not stop.stopped
    assert sorted(steps) == list(range(MAX_STEPS))


def epoch_end(stop, should_evaluate, epoch=2.0):
    control = TrainerControl(should_evaluate=should_evaluate)
    stop.on_epoch_end(
        SimpleNamespace(device=torch.device("cpu")), SimpleNamespace(epoch=epoch, global_step=8), control
    )
    return control


def test_stop_inside_an_epoch_does_not_evaluate(tmp_path):
    # Hugging Face ends the epoch loop after a stop, and 1.88 would round to an evaluated epoch.
    stop = GracefulStopCallback(tmp_path / STOP_FILENAME, end_time=time.time() + 5 * 3600)
    stop.stopped = True
    control = epoch_end(stop, should_evaluate=True, epoch=1.88)
    assert not control.should_evaluate


def test_evaluation_that_might_not_finish_is_skipped(tmp_path):
    stop = GracefulStopCallback(
        tmp_path / STOP_FILENAME, end_time=time.time() + 40 * 60,
        margin_seconds=20 * 60, evaluation_budget_seconds=30 * 60,
    )
    control = epoch_end(stop, should_evaluate=True)
    assert stop.stopped
    assert not control.should_evaluate
    assert control.should_save and control.should_training_stop
    # The job that resumes runs it.
    assert (tmp_path / PENDING_EVALUATION_FILENAME).read_text() == "8\n"


def test_requested_stop_still_evaluates_when_there_is_time(tmp_path):
    (tmp_path / STOP_FILENAME).touch()
    stop = GracefulStopCallback(tmp_path / STOP_FILENAME, end_time=time.time() + 5 * 3600)
    control = epoch_end(stop, should_evaluate=True)
    assert stop.stopped
    assert control.should_evaluate
    assert control.should_save and control.should_training_stop


def test_evaluation_budget_grows_to_the_longest_evaluation(tmp_path):
    stop = GracefulStopCallback(tmp_path / STOP_FILENAME, evaluation_budget_seconds=60)
    epoch_end(stop, should_evaluate=True)
    stop._evaluation_start -= 300
    stop.on_evaluate(None, None, TrainerControl())
    assert stop.evaluation_budget_seconds == pytest.approx(300, abs=5)


def test_slurm_end_time_is_read_from_the_environment_without_scontrol(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1793476500")

    def fail(*args, **kwargs):
        raise AssertionError("scontrol should not be called")

    monkeypatch.setattr(control_module.subprocess, "run", fail)
    assert control_module.slurm_job_end_time() == 1793476500


def test_slurm_end_time_is_read_from_scontrol(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    output = "JobId=123 JobName=run\n   StartTime=2026-10-31T09:35:00 EndTime=2026-10-31T21:35:00\n"
    monkeypatch.setattr(
        control_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=output),
    )
    expected = datetime.fromisoformat("2026-10-31T21:35:00").timestamp()
    assert control_module.slurm_job_end_time() == expected


def test_no_slurm_end_time_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert control_module.slurm_job_end_time() is None


def ddp_worker(rank, world_size, rendezvous, output_dir):
    torch.set_num_threads(1)
    AcceleratorState._reset_state(reset_partial_state=True)
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29502",
    )
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=Path(rendezvous).as_uri(),
        timeout=timedelta(seconds=60),
    )
    try:
        # Only rank 0 knows the time limit, as with 'from_environment'.
        stop = GracefulStopCallback(
            Path(output_dir, STOP_FILENAME),
            end_time=time.time() + 30 if rank == 0 else None,
            margin_seconds=60,
            check_every_n_steps=2,
        )
        model, _, steps = train(
            output_dir, world_size=world_size, callbacks=[stop], stop_requested=lambda: stop.stopped
        )
        assert stop.stopped
        assert sorted(steps) == [0, 1]
        parameters = flat_parameters(model)
        gathered = [torch.empty_like(parameters) for _ in range(world_size)]
        dist.all_gather(gathered, parameters)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU DDP requires Gloo")
def test_all_ranks_follow_rank_0_and_stop_at_the_same_step(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.multiprocessing.start_processes(
            ddp_worker,
            args=(2, str(tmp_path / "rendezvous"), str(tmp_path)),
            nprocs=2,
            start_method="spawn",
        )
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize(
    "run_inference, already_complete, markers, expected",
    [
        (True, False, [], True),
        (False, False, [], False),
        # A finished run from before these markers keeps its earlier inference.
        (True, True, [], False),
        # Training finished, but the job ended before inference finished.
        (True, True, ["TRAINING_COMPLETE"], True),
        (True, True, ["TRAINING_COMPLETE", "RUN_COMPLETE"], False),
    ],
)
def test_inference_runs_once_after_training(tmp_path, run_inference, already_complete, markers, expected):
    import train as train_script

    for marker in markers:
        (tmp_path / marker).touch()
    assert train_script.inference_is_pending(tmp_path, run_inference, already_complete) == expected


def test_slurm_end_time_is_retried_after_a_timeout(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    outputs = iter(["", "JobId=123 EndTime=2026-10-31T21:35:00"])
    monkeypatch.setattr(
        control_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, stdout=next(outputs)),
    )
    expected = datetime.fromisoformat("2026-10-31T21:35:00").timestamp()
    assert control_module.slurm_job_end_time(retry_seconds=0) == expected


def step_end(stop, step):
    control = TrainerControl()
    stop.on_step_end(SimpleNamespace(device=torch.device("cpu")), SimpleNamespace(global_step=step), control)
    return control


def test_resumed_job_runs_the_skipped_evaluation_once(tmp_path):
    (tmp_path / PENDING_EVALUATION_FILENAME).write_text("8\n")
    stop = GracefulStopCallback(tmp_path / STOP_FILENAME, end_time=time.time() + 5 * 3600)
    assert step_end(stop, 9).should_evaluate
    stop.on_evaluate(None, None, TrainerControl())
    assert not (tmp_path / PENDING_EVALUATION_FILENAME).exists()
    assert not step_end(stop, 10).should_evaluate


def test_resuming_before_the_skipped_evaluation_leaves_it_to_its_epoch_end(tmp_path):
    (tmp_path / PENDING_EVALUATION_FILENAME).write_text("8\n")
    stop = GracefulStopCallback(tmp_path / STOP_FILENAME, end_time=time.time() + 5 * 3600)
    assert not step_end(stop, 5).should_evaluate
    assert not (tmp_path / PENDING_EVALUATION_FILENAME).exists()


def test_evaluation_that_fits_is_not_skipped_with_the_default_budget(tmp_path, monkeypatch):
    # 41 minutes are left: enough for the 20-minute margin and the 20-minute evaluation budget.
    monkeypatch.delenv("VLM_STOP_EVALUATION_MINUTES", raising=False)
    monkeypatch.delenv("VLM_STOP_MARGIN_MINUTES", raising=False)
    monkeypatch.setattr(control_module, "slurm_job_end_time", lambda: time.time() + 41 * 60)
    stop = GracefulStopCallback.from_environment(tmp_path)
    assert stop.evaluation_budget_seconds == 20 * 60
    control = epoch_end(stop, should_evaluate=True)
    assert control.should_evaluate and not stop.stopped
    assert not (tmp_path / PENDING_EVALUATION_FILENAME).exists()
