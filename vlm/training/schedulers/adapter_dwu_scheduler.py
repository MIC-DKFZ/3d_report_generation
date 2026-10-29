from pathlib import Path

from torch.optim import Optimizer

from vlm.training.schedulers.base import SchedulerBase, TrainingPhase, cosine_decay


class AdapterDWUScheduler(SchedulerBase):
    """
    Double warmup: the adapter first trains alone, then all configured parameters warm up again.

    WARMUP_1 phase: only the adapter trains, with a linear warmup over the whole phase.
    DEFAULT phase: all configured parameters train with a fresh optimizer, a second linear
    warmup over 'warmup_2_duration', and a cosine decay.

    Durations are fractions of the whole training; WARMUP_1 is rounded to whole epochs.
    """

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
    ):
        super().__init__(
            optimizer_class,
            optimizer_kwargs,
            module_attr_map,
            forbidden_name_patterns_for_wd,
            output_dir,
        )
        self.lr_param_groups = [
            ("adapter", self.module_attr_map["adapter"], adapter_lr),
            ("default", "", lm_lr),
        ]
        self.warmup_1_duration = warmup_1_duration
        self.warmup_2_duration = warmup_2_duration

    def warmup_1_epochs(self, num_train_epochs: int) -> int:
        epochs = round(self.warmup_1_duration * num_train_epochs)
        if epochs >= num_train_epochs or (epochs == 0 and self.warmup_1_duration > 0):
            raise ValueError(
                f"The first warmup ({self.warmup_1_duration} of {num_train_epochs} epochs) must "
                "round to at least one epoch and leave at least one epoch after it."
            )
        return epochs

    def warmup_1_ramp_steps(self, max_steps: int, warmup_1_steps: int) -> int:
        """Steps of the linear warmup that starts WARMUP_1; the learning rate is constant after it."""
        return warmup_1_steps

    def phases(self, num_train_epochs: int) -> list[TrainingPhase]:
        warmup_1_epochs = self.warmup_1_epochs(num_train_epochs)
        default = TrainingPhase("DEFAULT", warmup_1_epochs, num_train_epochs)
        if warmup_1_epochs == 0:
            return [default]
        return [TrainingPhase("WARMUP_1", 0, warmup_1_epochs, adapter_only=True), default]

    def lr_schedule(self, max_steps: int, steps_per_epoch: int, num_train_epochs: int):
        warmup_1_steps = self.warmup_1_epochs(num_train_epochs) * steps_per_epoch
        warmup_1_ramp_steps = self.warmup_1_ramp_steps(max_steps, warmup_1_steps)
        warmup_2_steps = int(self.warmup_2_duration * max_steps)
        decay_start = warmup_1_steps + warmup_2_steps

        def lr(step: int, max_lr: float) -> float:
            if step < warmup_1_steps:
                return max_lr * min(1.0, (step + 1) / warmup_1_ramp_steps)
            if step < decay_start:
                return max_lr * (step - warmup_1_steps + 1) / warmup_2_steps
            return cosine_decay(step - decay_start, max_steps - decay_start, max_lr)

        return lr
