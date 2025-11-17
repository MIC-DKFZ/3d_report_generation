import torch.nn as nn
from typing import Literal
from pydantic import Field
from configs.base import ConfigBase
from configs.model.adapter.adapter_config import AdapterDynamicClassConfig
from configs.model.trainable_params.trainable_params_config import TrainableParamsConfig
from configs.model.vision_encoder_config import VisionEncoderConfig

class ModelConfig(ConfigBase):
    model_type: Literal["owl", "qwen"] = Field(..., description="Type of VLM model to train.")
    model_id: str | None = Field(None, description="Model ID or path.")

    vision_encoder: VisionEncoderConfig | None = Field(None, description="Should only be None, if you train with embeddings instead of images. "
                                                                         "or have text-only inputs.")
    adapter: AdapterDynamicClassConfig | list[AdapterDynamicClassConfig] | None \
        = Field(None, description="Should only be None, if you have text-only inputs.")

    trainable_params: TrainableParamsConfig

    def build_adapter(self) -> nn.Module:
        if self.adapter is None:
            raise ValueError(f"Cannot build adapter with 'adapter=None'")

        if isinstance(self.adapter, list):
            return nn.Sequential(*[a.to_instance() for a in self.adapter])
        return self.adapter.to_instance()