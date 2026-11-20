"""A stop request or a close Slurm time limit saves and stops, and a follow-up job continues."""

import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate.state import AcceleratorState
from test_training_phases import MAX_STEPS, flat_parameters, train, uninterrupted  # noqa: F401
from transformers import TrainerCallback

from vlm.training.callbacks import STOP_FILENAME, GracefulStopCallback


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
