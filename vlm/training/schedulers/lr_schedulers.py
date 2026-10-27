import math
from typing import List
from torch.optim.lr_scheduler import LRScheduler


class ConstantLR(LRScheduler):
    def __init__(self, optimizer, param_groups: list[dict], current_step: int = None, **kwargs):
        self.optimizer = optimizer
        self.max_lrs = [param_group["lr"] for param_group in param_groups]
        self.ctr = 0
        self.has_stepped_once = False
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            self.ctr += 1

        if not self.has_stepped_once:
            self.has_stepped_once = True

            for param_group, max_lr in zip(self.optimizer.param_groups, self.max_lrs):
                param_group["lr"] = max_lr

            self._last_lr: List[float] = [group["lr"] for group in self.optimizer.param_groups]



class LinearIncreasingLR(LRScheduler):
    def __init__(self, optimizer, param_groups: list[dict], max_steps: int, current_step: int = None):
        self.optimizer = optimizer
        self.max_lrs = [param_group["lr"] for param_group in param_groups]
        self.max_steps = max_steps
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        for param_group, max_lr in zip(self.optimizer.param_groups, self.max_lrs):
            new_lr = max_lr / self.max_steps * (1 + current_step)
            param_group["lr"] = new_lr

        self._last_lr: List[float] = [group["lr"] for group in self.optimizer.param_groups]


class LinearIncreasingOffsetLR(LRScheduler):
    def __init__(self, optimizer, param_groups: list[dict], max_steps: int, start_step: int, current_step: int = None):
        self.optimizer = optimizer
        self.max_lrs = [param_group["lr"] for param_group in param_groups]
        self.max_steps = max_steps - start_step
        self.start_step = start_step
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        current_step = current_step - self.start_step
        if current_step <= 0:
            current_step = 0

        for param_group, max_lr in zip(self.optimizer.param_groups, self.max_lrs):
            new_lr = max_lr / self.max_steps * (1 + current_step)
            param_group["lr"] = new_lr

        self._last_lr: List[float] = [group["lr"] for group in self.optimizer.param_groups]


class PolyOffsetLR(LRScheduler):
    def __init__(
            self,
            optimizer,
            param_groups: list[dict],
            max_steps: int,
            start_step: int,
            exponent: float = 0.9,
            current_step: int = None,
    ):
        self.optimizer = optimizer
        self.max_lrs = [param_group["lr"] for param_group in param_groups]
        self.max_steps = max_steps - start_step
        self.start_step = start_step
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        current_step = current_step - self.start_step
        if current_step <= 0:
            current_step = 0

        for param_group, max_lr in zip(self.optimizer.param_groups, self.max_lrs):
            new_lr = max_lr * (1 - current_step / self.max_steps) ** self.exponent
            param_group["lr"] = new_lr

        self._last_lr: List[float] = [group["lr"] for group in self.optimizer.param_groups]


class CosineAnnealingOffsetLR(LRScheduler):
    def __init__(
            self,
            optimizer,
            param_groups: list[dict],
            max_steps: int,
            start_step: int,
            current_step: int = None,
            min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.max_lrs = [param_group["lr"] for param_group in param_groups]
        self.max_steps = max_steps - start_step
        self.start_step = start_step
        self.min_lr = min_lr
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        current_step = current_step - self.start_step
        if current_step <= 0:
            current_step = 0

        for param_group, max_lr in zip(self.optimizer.param_groups, self.max_lrs):
            new_lr = (self.min_lr
                      + (max_lr - self.min_lr)
                      * (1 + math.cos(math.pi * current_step / self.max_steps))
                      / 2)
            param_group["lr"] = new_lr

        self._last_lr: List[float] = [group["lr"] for group in self.optimizer.param_groups]
