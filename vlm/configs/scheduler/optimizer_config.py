from vlm.configs.base import RegistryClassConfigBase
from vlm.registries.optimizer_registry import OPTIMIZER_REGISTRY


class OptimizerClassConfig(RegistryClassConfigBase):
    registry = OPTIMIZER_REGISTRY