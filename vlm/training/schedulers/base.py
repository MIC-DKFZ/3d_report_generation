import os
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from transformers.trainer import SCHEDULER_NAME, OPTIMIZER_NAME

from vlm.utils import logger
from vlm.training.checkpoint_state import record_trainable_checkpoint_keys
from vlm.utils.helper import log_param_groups, is_main_process


class SchedulerBase(ABC):
    def __init__(
            self,
            optimizer_class: type[Optimizer],
            optimizer_kwargs: dict,
            module_attr_map: dict,
            forbidden_name_patterns_for_wd: list[str],
            output_dir: str | Path
    ):
        self.training_stage: StrEnum | None = None
        self.new_training_stage: bool = False

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.module_attr_map = module_attr_map
        self.forbidden_name_patterns_for_wd = forbidden_name_patterns_for_wd
        self.output_dir = output_dir

        self.orig_trainable_params_state = {}
        self._is_initialized = False

    def initialize(self, model: nn.Module, max_steps: int):
        if self._is_initialized:
            logger.warning_main("Scheduler has already been initialized. Skipping repeated initialization.")
            return

        for name, param in model.named_parameters():
            self.orig_trainable_params_state[name] = param.requires_grad

        self._initialize(max_steps=max_steps)
        self._is_initialized = True

    def _restore_orig_trainable_params_state(self, model: nn.Module):
        """
        Restore the original `requires_grad` state of parameters recorded during `self.initialize()`.
        """
        for name, param in model.named_parameters():
            if name in self.orig_trainable_params_state:
                param.requires_grad = self.orig_trainable_params_state[name]

    def _log_param_groups(self, model: nn.Module, optimizer: Optimizer):
        # file_path = Path(self.output_dir, f"param_groups_{self.training_stage}_{os.getenv("LOCAL_RANK")}.log")
        file_path = Path(self.output_dir, f"param_groups__{self.training_stage}.log")
        log_param_groups(model, optimizer, file_path)

    def prepare_training_step(
            self,
            model,
            optimizer,
            lr_scheduler,
            global_step: int,
            max_steps: int,
    ) -> tuple[Optimizer, _LRScheduler, bool]:
        """Configure trainability, the optimizer, and the LR scheduler for a training step."""
        self.update_training_stage(global_step=global_step, max_steps=max_steps)
        changed_requires_grad_status = self.set_trainable_params(model=model)
        if changed_requires_grad_status:
            record_trainable_checkpoint_keys(model)
        optimizer, lr_scheduler = self.configure_optimizer_and_lr_scheduler(
            model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
        if self.new_training_stage:
            if is_main_process():
                self._log_param_groups(model, optimizer)
            self.new_training_stage = False

        return optimizer, lr_scheduler, changed_requires_grad_status

    def restore_checkpoint_state(
            self,
            model,
            optimizer,
            lr_scheduler,
            global_step: int,
            max_steps: int,
            checkpoint_dir: str | Path,
    ) -> tuple[Optimizer, _LRScheduler, bool]:
        """Restore the optimizer and scheduler state that produced a checkpoint."""
        checkpoint_stage_step = max(global_step - 1, 0)
        optimizer, lr_scheduler, changed_requires_grad_status = self.prepare_training_step(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            global_step=checkpoint_stage_step,
            max_steps=max_steps,
        )

        scheduler_state = torch.load(
            os.path.join(checkpoint_dir, SCHEDULER_NAME),
            weights_only=True,
        )
        lr_scheduler.load_state_dict(scheduler_state)
        optimizer_state = torch.load(
            os.path.join(checkpoint_dir, OPTIMIZER_NAME),
            map_location=model.device,
            weights_only=True,
        )
        optimizer.load_state_dict(optimizer_state)

        return optimizer, lr_scheduler, changed_requires_grad_status

    def resume(
            self,
            model,
            optimizer,
            lr_scheduler,
            global_step: int,
            max_steps: int,
            checkpoint_dir: str,
    ) -> tuple[Optimizer, _LRScheduler, bool]:
        """
        Restore the optimizer/scheduler state and prepare the next training step.

        Trainer increments ``global_step`` after an optimizer step and saves the
        checkpoint afterwards. A checkpoint at step N therefore contains state
        produced by the scheduler configuration for step N - 1, while resumed
        training must use the configuration for step N.
        """
        logger.info_main(
            "[SCHEDULER] Restoring checkpoint state from the stage active at "
            f"step {max(global_step - 1, 0)}, then preparing step {global_step}."
        )

        # Recreate the stage that produced the checkpoint before loading its state.
        optimizer, lr_scheduler, changed_for_checkpoint_stage = self.restore_checkpoint_state(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            global_step=global_step,
            max_steps=max_steps,
            checkpoint_dir=checkpoint_dir,
        )
        # Advance the scheduler configuration to the stage used by the next step.
        optimizer, lr_scheduler, changed_for_next_stage = self.prepare_training_step(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            global_step=global_step,
            max_steps=max_steps,
        )

        return (
            optimizer,
            lr_scheduler,
            changed_for_checkpoint_stage or changed_for_next_stage,
        )

    @abstractmethod
    def _initialize(self, max_steps):
        """
        Initialize stage boundaries using `max_steps` provided by the Trainer.
        """
        ...

    @abstractmethod
    def update_training_stage(self, global_step: int, max_steps: int):
        """
        Update the internal training stage based on the current progress.

        Implementations should decide which stage is active given the global
        step, update the stored stage in `self.training_stage`, and set
        `self.new_training_stage=True` when a stage transition occurs
        so the other methods can react accordingly.
        """
        ...

    @abstractmethod
    def configure_optimizer_and_lr_scheduler(
            self,
            model,
            optimizer: Optimizer | None,
            lr_scheduler: _LRScheduler
    ) -> tuple[Optimizer, _LRScheduler]:
        """
        Reconfigure the optimizer and LR scheduler when entering a new training stage.

        If `self.new_training_stage` is set, this method may recreate the optimizer
        (e.g., to reset momentum) and will initialize the appropriate scheduler
        for the current training stage. Returns the updated optimizer and scheduler.
        """
        ...

    @abstractmethod
    def set_trainable_params(self, model) -> bool:
        """
        Control which parameters are trainable for each training stage.

        This method is responsible for enabling or disabling gradient updates
        (via the `requires_grad` flag) for specific subsets of the model's
        parameters. It is typically called at the beginning of a training
        stage or whenever the training configuration changes.

        Args:
            model: A Hugging Face Transformers model whose parameters may be
                selectively frozen or unfrozen.

        Returns:
            changed_requires_grad_state (bool): True if this method modified
            the `requires_grad` state of **any** parameter. This signals to
            `VLMSFTTrainer` that the model must be re-wrapped in Distributed
            Data Parallel (DDP), because changing which parameters require
            gradients alters the gradient buckets.
        """
        ...
