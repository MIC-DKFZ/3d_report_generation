from vlm.configs.base import RecursiveFindClassConfigBase
from vlm.configs.scheduler.optimizer_config import OptimizerClassConfig
from vlm.training.schedulers.base import SchedulerBase


class SchedulerClassConfig(RecursiveFindClassConfigBase):
    """
    This config controls the usage of our custom schedulers defined at 'vlm.training.schedulers'.
    """
    folder_to_search = "vlm.training.schedulers"

    optimizer: OptimizerClassConfig

    def to_instance(self, **kwargs) -> SchedulerBase:
        return self._resolved_class(
            optimizer_class=self.optimizer._resolved_class,
            optimizer_kwargs=self.optimizer.class_kwargs,
            **self.class_kwargs,
            **kwargs
        )

