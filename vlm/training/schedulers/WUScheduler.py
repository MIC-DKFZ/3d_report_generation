from abc import ABC
from enum import StrEnum
from pathlib import Path
from vlm.training.schedulers.lr_schedulers import LinearIncreasingLR, CosineAnnealingOffsetLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from vlm.training.schedulers.base import SchedulerBase
from vlm.utils import logger
from vlm.utils.optimizer_and_params import get_param_groups


class WUScheduler(SchedulerBase, ABC):

    class _Stage(StrEnum):
        WARMUP = "WARMUP"
        DEFAULT = "DEFAULT"

    def __init__(
            self,
            optimizer_class: type[Optimizer],
            optimizer_kwargs: dict,
            module_attr_map: dict,
            forbidden_name_patterns_for_wd: list[str],
            output_dir: str | Path,

            adapter_lr: float,  # This should be called 'lr' as this learning rate is applied to all trainable modules.
                                # I leave it as is, but this can be safely renamed in the future.
            warmup_duration: float
    ):
        super().__init__(optimizer_class, optimizer_kwargs, module_attr_map, forbidden_name_patterns_for_wd, output_dir)

        self.lr_param_groups = [
            ("default", "", adapter_lr)
        ]

        self.warmup_duration = warmup_duration

        self.warmup_max_step = None
        self.max_steps = None

    def _initialize(self, max_steps: int):
        self.max_steps = max_steps
        self.warmup_max_step = int(self.warmup_duration * self.max_steps)

    def update_training_stage(self, global_step: int, max_steps: int):
        if global_step < self.warmup_max_step:
            next_training_stage = self._Stage.WARMUP
        else:
            next_training_stage = self._Stage.DEFAULT

        if self.training_stage != next_training_stage:
            self.training_stage = next_training_stage
            self.new_training_stage = True

    def set_trainable_params(self, model) -> bool:
        return False

    def configure_optimizer_and_lr_scheduler(
            self,
            model,
            optimizer: Optimizer | None,
            lr_scheduler: LRScheduler
    ) -> tuple[Optimizer, LRScheduler]:
        if not self.new_training_stage:
            return optimizer, lr_scheduler

        # These parameter groups always contain the maximum learning rates for each group, which need to
        # be known by the learning rate scheduler.
        param_groups = get_param_groups(
            self.lr_param_groups,
            self.optimizer_kwargs.get("weight_decay", 0.0),
            model,
            self.forbidden_name_patterns_for_wd
        )

        if optimizer is None:
            optimizer = self.optimizer_class(param_groups, **self.optimizer_kwargs)

        if self.training_stage == self._Stage.WARMUP:
            logger.info_main("[SCHEDULER] Warmup.")
            lr_scheduler = LinearIncreasingLR(optimizer=optimizer,
                                              param_groups=param_groups,
                                              max_steps=self.warmup_max_step)
        elif self.training_stage == self._Stage.DEFAULT:
            logger.info_main("[SCHEDULER] Normal Training.")
            lr_scheduler = CosineAnnealingOffsetLR(optimizer=optimizer,
                                                   param_groups=param_groups,
                                                   max_steps=self.max_steps,
                                                   start_step=self.warmup_max_step)
        else:
            raise RuntimeError(f"unknown training stage: {self.training_stage}")

        logger.info_main(
            f"[SCHEDULER] Initialized {self.training_stage} lr_scheduler ({type(lr_scheduler).__name__})")

        return optimizer, lr_scheduler
