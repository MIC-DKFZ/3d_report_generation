from vlm.configs.base import ConfigBase
from vlm.configs.model.trainable_params.trainable_linear_layer import TrainableLinearConfig
from vlm.configs.model.trainable_params.trainable_module import TrainableModuleConfig


class TrainableParamsConfig(ConfigBase):
    vision_encoder: TrainableModuleConfig | None = None
    adapter: TrainableModuleConfig | None = None
    language_model: TrainableModuleConfig
    lm_head: TrainableLinearConfig