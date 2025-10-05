from configs.base import ConfigBase
from configs.model.trainable_params.trainable_linear_layer import TrainableLinearConfig
from configs.model.trainable_params.trainable_module import TrainableModuleConfig


class TrainableParamsConfig(ConfigBase):
    vision_encoder: TrainableModuleConfig | None = None
    adapter: TrainableModuleConfig
    language_model: TrainableModuleConfig
    lm_head: TrainableLinearConfig