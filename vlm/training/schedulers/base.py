import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from vlm.training.parameters import get_param_groups, log_param_groups
from vlm.utils.ddp import is_main_process

MIN_LR = 1e-6


@dataclass(frozen=True)
class TrainingPhase:
    """Whole epochs trained with one set of trainable parameters and one optimizer."""

    name: str
    start_epoch: int
    end_epoch: int
    adapter_only: bool = False


def cosine_decay(step: int, steps: int, max_lr: float) -> float:
    return MIN_LR + (max_lr - MIN_LR) * (1 + math.cos(math.pi * step / steps)) / 2


class SchedulerBase(ABC):
    """
    Split training into phases and define the learning rate of every optimizer step.

    Each phase is trained by its own 'VLMTrainer', so changing the trainable parameters always
    starts a fresh optimizer. The learning rate is a function of the global step over the whole
    run, so a later phase continues the schedule where the previous phase ended.
    """

    def __init__(
        self,
        optimizer_class: type[Optimizer],
        optimizer_kwargs: dict,
        module_attr_map: dict,
        forbidden_name_patterns_for_wd: list[str],
        output_dir: str | Path,
    ):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.module_attr_map = module_attr_map
        self.forbidden_name_patterns_for_wd = forbidden_name_patterns_for_wd
        self.output_dir = output_dir

        # (module name, parameter name prefix, learning rate); earlier entries take precedence.
        self.lr_param_groups: list[tuple[str, str, float]] = []
        self.configured_requires_grad: dict[str, bool] | None = None

    @abstractmethod
    def phases(self, num_train_epochs: int) -> list[TrainingPhase]:
        """Return consecutive phases that cover all training epochs."""

    @abstractmethod
    def lr_schedule(
        self, max_steps: int, steps_per_epoch: int, num_train_epochs: int
    ) -> Callable[[int, float], float]:
        """Return 'lr(step, max_lr)': the learning rate of a group at a global step."""

    def set_trainable_params(self, model: nn.Module, phase: TrainingPhase):
        # The first call sees the flags set by the model config and PEFT, before any phase.
        if self.configured_requires_grad is None:
            self.configured_requires_grad = {
                name: param.requires_grad for name, param in model.named_parameters()
            }
        adapter_prefix = self.module_attr_map["adapter"]
        for name, param in model.named_parameters():
            param.requires_grad = self.configured_requires_grad[name] and (
                not phase.adapter_only or name.startswith(adapter_prefix)
            )

    def create_optimizer(self, model: nn.Module, phase: TrainingPhase) -> Optimizer:
        param_groups = get_param_groups(
            self.lr_param_groups,
            self.optimizer_kwargs.get("weight_decay", 0.0),
            model,
            self.forbidden_name_patterns_for_wd,
        )
        optimizer = self.optimizer_class(param_groups, **self.optimizer_kwargs)
        if is_main_process():
            log_param_groups(
                model, optimizer, Path(self.output_dir, f"param_groups__{phase.name}.log")
            )
        return optimizer

    def create_lr_scheduler(
        self,
        optimizer: Optimizer,
        max_steps: int,
        steps_per_epoch: int,
        num_train_epochs: int,
        start_step: int,
    ) -> LambdaLR:
        lr = self.lr_schedule(max_steps, steps_per_epoch, num_train_epochs)
        # LambdaLR scales each group's configured learning rate by the returned factor.
        lr_lambdas = [
            lambda step, max_lr=group["lr"]: lr(step, max_lr) / max_lr if max_lr else 0.0
            for group in optimizer.param_groups
        ]
        # Starting at 'start_step' requires 'initial_lr', which LambdaLR only sets at step 0.
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        return LambdaLR(optimizer, lr_lambdas, last_epoch=start_step - 1)
