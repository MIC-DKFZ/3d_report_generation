from abc import ABC
from enum import StrEnum
from pathlib import Path

from vlm.training.schedulers.lr_schedulers import LinearIncreasingLR, LinearIncreasingOffsetLR, \
    CosineAnnealingOffsetLR, ConstantLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from vlm.training.schedulers.base import SchedulerBase
from vlm.utils import logger


class AdapterDWUScheduler(SchedulerBase, ABC):

    class _Stage(StrEnum):
        WARMUP_1 = "WARMUP_1"
        WARMUP_2 = "WARMUP_2"
        DEFAULT = "DEFAULT"

    def __init__(
            self,
            optimizer_class: type[Optimizer],
            optimizer_kwargs: dict,
            module_attr_map: dict,
            forbidden_name_patterns_for_wd: list[str],
            output_dir: str | Path,

            adapter_lr: float,
            lm_lr: float,
            warmup_1_duration: float,
            warmup_2_duration: float,
            reset_optimizer: bool = True
    ):
        super().__init__(optimizer_class, optimizer_kwargs, module_attr_map, forbidden_name_patterns_for_wd, output_dir)

        self.lr_param_groups = [
            ("adapter", self.module_attr_map["adapter"], adapter_lr),
            ("default", "", lm_lr)  # Set the global default, order matters!
        ]

        self.warmup_1_duration = warmup_1_duration
        self.warmup_2_duration = warmup_2_duration
        self.reset_optimizer = reset_optimizer

        self.lr_scheduler_wu_1 = LinearIncreasingLR
        self.lr_scheduler_wu_2 = LinearIncreasingOffsetLR
        self.lr_scheduler_default = CosineAnnealingOffsetLR

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
            changed = False
            for name, param in model.named_parameters():
                if not name.startswith(self.module_attr_map["adapter"]):
                    if param.requires_grad:
                        changed = True
                        param.requires_grad = False
            return changed
        elif self.training_stage in {self._Stage.WARMUP_2, self._Stage.DEFAULT}:
            return self._restore_configured_trainable_params_state(model)
        else:
            raise RuntimeError(f"Unknown training stage: {self.training_stage}")

    def configure_optimizer_and_lr_scheduler(
            self,
            model,
            optimizer: Optimizer | None,
            lr_scheduler: LRScheduler
    ) -> tuple[Optimizer, LRScheduler]:
        if not self.new_training_stage:
            return optimizer, lr_scheduler

        optimizer, param_groups = self._configure_optimizer(
            model,
            optimizer,
            reset=(self.training_stage == self._Stage.WARMUP_2 and self.reset_optimizer),
        )

        if self.training_stage == self._Stage.WARMUP_1:
            logger.info_main("[SCHEDULER] Adapter-Only Training. First Warmup.")
            lr_scheduler = self.lr_scheduler_wu_1(optimizer=optimizer,
                                                  param_groups=param_groups,
                                                  max_steps=self.warmup_1_max_step)
        elif self.training_stage == self._Stage.WARMUP_2:
            logger.info_main("[SCHEDULER] Normal Training. Second Warmup.")
            lr_scheduler = self.lr_scheduler_wu_2(optimizer=optimizer,
                                                  param_groups=param_groups,
                                                  max_steps=self.warmup_2_max_step,
                                                  start_step=self.warmup_1_max_step)
        elif self.training_stage == self._Stage.DEFAULT:
            logger.info_main("[SCHEDULER] Normal Training. Warmups Done.")
            lr_scheduler = self.lr_scheduler_default(optimizer=optimizer,
                                                     param_groups=param_groups,
                                                     max_steps=self.max_steps,
                                                     start_step=self.warmup_2_max_step)
        else:
            raise RuntimeError(f"unknown training stage: {self.training_stage}")

        logger.info_main(
            f"[SCHEDULER] Initialized {self.training_stage} lr_scheduler ({type(lr_scheduler).__name__})")

        return optimizer, lr_scheduler


class AdapterDWUConstScheduler(AdapterDWUScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler_default = ConstantLR
