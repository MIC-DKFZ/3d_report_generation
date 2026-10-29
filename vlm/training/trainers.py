import os

import torch
from transformers import Trainer, TrainerState
from transformers.trainer import SCHEDULER_NAME

from vlm.training.checkpoint_state import load_model_checkpoint, save_model_checkpoint
from vlm.training.schedulers.base import SchedulerBase, TrainingPhase
from vlm.utils import logger


class VLMTrainer(Trainer):
    """
    Trainer for one training phase: a fixed set of trainable parameters and one optimizer.

    The scheduler builds the optimizer and the learning-rate schedule of the whole run. A phase
    that starts later continues from 'phase_start_state', the trainer state at the end of the
    previous phase, with a fresh optimizer.
    """

    def __init__(
        self,
        scheduler: SchedulerBase,
        phase: TrainingPhase,
        phase_start_state: TrainerState | None = None,
        model_weights_preloaded: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scheduler = scheduler
        self.phase = phase
        self.phase_start_state = phase_start_state
        self.model_weights_preloaded = model_weights_preloaded
        self.steps_per_epoch = None

    def _get_num_items_in_batch(self, batch_samples, device):
        num_items_in_batch = super()._get_num_items_in_batch(batch_samples, device)
        if num_items_in_batch is not None and num_items_in_batch.item() == 0:
            raise ValueError("The accumulation window contains no supervised tokens.")
        return num_items_in_batch

    def log(self, logs, start_time=None):
        # WandbGroupedLRCallback logs the learning rate of every parameter group instead.
        logs.pop("learning_rate", None)
        if "loss" in logs and torch.cuda.is_available():
            # Peak since the previous training log, then reset so each interval shows its own peak.
            logs["gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            logs["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
            torch.cuda.reset_peak_memory_stats()
        super().log(logs, start_time)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if self.model_weights_preloaded:
            logger.info_main(
                "Skipping Trainer model-weight loading because checkpoint weights were preloaded."
            )
            return

        load_model_checkpoint(model if model is not None else self.model, resume_from_checkpoint)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        model = self.accelerator.unwrap_model(self.model)
        save_model_checkpoint(model, output_dir, state_dict=state_dict)

    def _save_optimizer_and_scheduler(self, output_dir):
        if not self.is_deepspeed_enabled:
            return super()._save_optimizer_and_scheduler(output_dir)
        # Frozen weights come from the pretrained model and trained ones from 'model.safetensors',
        # so resuming only needs DeepSpeed's optimizer state.
        self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
        if self.args.should_save:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        if not self.is_deepspeed_enabled or resume_from_checkpoint is None:
            return super()._prepare_for_training(
                max_steps, train_dataloader, resume_from_checkpoint
            )
        # Upstream loads DeepSpeed checkpoints strictly, which fails without the frozen weights.
        model, train_dataloader = super()._prepare_for_training(max_steps, train_dataloader, None)
        load_path, _ = self.model_wrapped.load_checkpoint(
            resume_from_checkpoint, load_module_strict=False
        )
        if load_path is None:
            raise ValueError(f"DeepSpeed failed to resume from {resume_from_checkpoint}")
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)
        return model, train_dataloader

    def _init_training_state(
        self,
        max_steps,
        num_update_steps_per_epoch,
        num_train_epochs,
        resume_from_checkpoint,
        trial,
    ):
        epochs_trained, steps_trained_in_current_epoch = super()._init_training_state(
            max_steps, num_update_steps_per_epoch, num_train_epochs, resume_from_checkpoint, trial
        )
        self.state.train_batch_size = self._train_batch_size
        self.steps_per_epoch = num_update_steps_per_epoch

        if self.phase_start_state is not None:
            # Continue the run where the previous phase ended; only the optimizer starts fresh.
            self.state = self.phase_start_state
            self.state.init_training_references(self, max_steps, num_train_epochs, trial)
            epochs_trained, steps_into_epoch = divmod(
                self.state.global_step, num_update_steps_per_epoch
            )
            if steps_into_epoch:
                raise ValueError(
                    "A training phase must start at an epoch boundary, not at global step "
                    f"{self.state.global_step}."
                )
        return epochs_trained, steps_trained_in_current_epoch

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            self.optimizer = self.scheduler.create_optimizer(self.model, self.phase)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = self.scheduler.create_lr_scheduler(
                optimizer or self.optimizer,
                max_steps=num_training_steps,
                steps_per_epoch=self.steps_per_epoch,
                num_train_epochs=self.state.num_train_epochs,
                start_step=self.state.global_step,
            )
        return self.lr_scheduler
