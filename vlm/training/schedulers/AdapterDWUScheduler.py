from abc import ABC
from enum import StrEnum
from pathlib import Path

from vlm.training.schedulers.lr_schedulers import Lin_incr_LRScheduler, Lin_incr_offset_LRScheduler, \
    CosineAnnealingLR_offset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from vlm.training.schedulers.base import SchedulerBase
from vlm.utils import logger
from vlm.utils.optimizer_and_params import create_optimizer_with_grouped_params


class AdapterDWUScheduler(SchedulerBase, ABC):

    class _Stage(StrEnum):
        WARMUP_1 = "WARMUP_1"
        WARMUP_2 = "WARMUP_2"
        DEFAULT = "DEFAULT"

    def __init__(
            self,
            optimizer_class: type[Optimizer],
            optimizer_kwargs: dict,
            warmup_1_duration: float,
            warmup_2_duration: float,

            module_attr_map: dict,
            forbidden_name_patterns_for_wd: list[str],
            output_dir: str | Path
    ):
        super().__init__(optimizer_class, optimizer_kwargs, module_attr_map, forbidden_name_patterns_for_wd, output_dir)

        self.warmup_1_duration = warmup_1_duration
        self.warmup_2_duration = warmup_2_duration

        self.warmup_1_max_step = None
        self.warmup_2_max_step = None
        self.max_steps = None

    def _initialize(self, max_steps: int):
        self.max_steps = max_steps
        self.warmup_1_max_step = int(self.warmup_1_duration * self.max_steps)
        self.warmup_2_max_step = int((self.warmup_1_duration + self.warmup_2_duration) * self.max_steps)

    def update_training_stage(self, global_step: int, max_steps: int):
        if global_step < self.warmup_1_max_step:
            next_training_stage = self._Stage.WARMUP_1
        elif global_step < self.warmup_2_max_step:
            next_training_stage = self._Stage.WARMUP_2
        else:
            next_training_stage = self._Stage.DEFAULT

        if self.training_stage != next_training_stage:
            self.training_stage = next_training_stage
            self.new_training_stage = True

    def set_trainable_params(self, model) -> bool:
        if not self.new_training_stage:
            return False

        if self.training_stage == self._Stage.WARMUP_1:
            for name, param in model.named_parameters():
                if not name.startswith(self.module_attr_map["adapter"]):
                    param.requires_grad = False
            return True
        elif self.training_stage == self._Stage.WARMUP_2:
            self._restore_orig_trainable_params_state(model)
            return True
        else: # if self.training_stage == self._Stage.DEFAULT:
            pass

        return False

    def configure_optimizer_and_lr_scheduler(
            self,
            model,
            optimizer: Optimizer | None,
            lr_scheduler: _LRScheduler
    ) -> tuple[Optimizer, _LRScheduler]:
        if not self.new_training_stage:
            return optimizer, lr_scheduler

        # If we use a momentum-based optimizer (e.g. AdamW), we do not want to keep the momentum on entering
        # 'WARMUP_2' from 'WARMUP_1'.
        # The newly unfrozen parameters start with zero momentum, thus adapting more slowly than the
        # decoder parameters, which could hinder overall training dynamics in the second warm-up phase.
        if optimizer is None or self.training_stage == self._Stage.WARMUP_2:
            if optimizer is not None:
                del optimizer
            optimizer = create_optimizer_with_grouped_params(
                self.optimizer_class,
                self.optimizer_kwargs,
                model,
                self.forbidden_name_patterns_for_wd
            )

        if self.training_stage == self._Stage.WARMUP_1:
            logger.info_main("[SCHEDULER] Adapter-Only Training. First Warmup.")
            lr_scheduler = Lin_incr_LRScheduler(optimizer=optimizer,
                                                     max_lr=self.max_lr,
                                                     max_steps=self.warmup_1_max_step)
        elif self.training_stage == self._Stage.WARMUP_2:
            logger.info_main("[SCHEDULER] Normal Training. Second Warmup.")
            lr_scheduler = Lin_incr_offset_LRScheduler(optimizer=optimizer,
                                                     max_lr=self.max_lr,
                                                     max_steps=self.warmup_2_max_step,
                                                     start_step=self.warmup_1_max_step)
        elif self.training_stage == self._Stage.DEFAULT:
            logger.info_main("[SCHEDULER] Normal Training. Warmups Done.")
            lr_scheduler = CosineAnnealingLR_offset(optimizer, T_max=self.max_steps, offset=self.warmup_2_max_step)
        else:
            raise RuntimeError(f"unknown training stage: {self.training_stage}")

        logger.info_main(
            f"[SCHEDULER] Initialized {self.training_stage} lr_scheduler ({type(lr_scheduler).__name__})")

        return optimizer, lr_scheduler
