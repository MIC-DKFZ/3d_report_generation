"""Train a run as consecutive phases, one 'VLMTrainer' per phase."""

from collections.abc import Callable
from pathlib import Path

from accelerate.state import AcceleratorState
from transformers import TrainerState
from transformers.trainer import TRAINER_STATE_NAME

from vlm.training.callbacks import PhaseEndCallback
from vlm.training.schedulers.base import TrainingPhase
from vlm.training.trainers import VLMTrainer


def train_phases(
    phases: list[TrainingPhase],
    create_trainer: Callable[[TrainingPhase, TrainerState | None], VLMTrainer],
    resume_checkpoint: str | Path | None = None,
    stop_requested: Callable[[], bool] = lambda: False,
) -> bool:
    """
    Train the phases left after 'resume_checkpoint'; return False if the run was already complete.

    'stop_requested' ends the run after the current phase, also when it stopped at the phase's
    last step, where the epoch alone would let the next phase start.

    'create_trainer(phase, phase_start_state)' configures the model for the phase and returns its
    trainer. Resuming inside a phase restores the phase's optimizer from the checkpoint; a phase
    that starts at the checkpoint or after the previous phase continues from the trainer state
    with a fresh optimizer.
    """
    state = None
    if resume_checkpoint is not None:
        state = TrainerState.load_from_json(str(Path(resume_checkpoint, TRAINER_STATE_NAME)))
        if state.global_step >= state.max_steps:
            return False

    for phase in phases:
        epoch = state.epoch if state is not None else 0
        if epoch >= phase.end_epoch:
            continue
        resume_inside_phase = resume_checkpoint is not None and epoch > phase.start_epoch
        trainer = create_trainer(phase, None if resume_inside_phase else state)
        if phase is not phases[-1]:
            # Added last, so no other callback can cancel the save at the end of the phase.
            trainer.add_callback(PhaseEndCallback(phase.end_epoch))
        trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_inside_phase else None
        )

        state, resume_checkpoint = trainer.state, None
        trainer.accelerator.free_memory()
        del trainer
        # The next phase's Trainer creates a new Accelerator with the same DeepSpeed plugin, which
        # accelerate only accepts once its shared state is cleared; the process group is kept.
        AcceleratorState._reset_state()
        if state.epoch < phase.end_epoch or stop_requested():
            break  # Early stopping ends the run, not just the phase.
    return True
