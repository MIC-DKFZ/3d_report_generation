from pathlib import Path

from torch.optim import Optimizer

from vlm.training.schedulers.adapter_dwu_scheduler import AdapterDWUScheduler


class AdapterExtendedDWUScheduler(AdapterDWUScheduler):
    """
    'AdapterDWUScheduler' whose adapter-only phase warms up over 'warmup_1_1_duration' and then
    keeps the learning rate constant for 'warmup_1_2_duration'.
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
        warmup_1_1_duration: float,
        warmup_1_2_duration: float,
        warmup_2_duration: float,
    ):
        super().__init__(
            optimizer_class,
            optimizer_kwargs,
            module_attr_map,
            forbidden_name_patterns_for_wd,
            output_dir,
            adapter_lr=adapter_lr,
            lm_lr=lm_lr,
            warmup_1_duration=warmup_1_1_duration + warmup_1_2_duration,
            warmup_2_duration=warmup_2_duration,
        )
        self.warmup_1_1_duration = warmup_1_1_duration

    def warmup_1_ramp_steps(self, max_steps: int, warmup_1_steps: int) -> int:
        return max(1, int(self.warmup_1_1_duration * max_steps))
