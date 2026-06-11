from pathlib import Path
from typing import Literal
from pydantic import Field

from vlm.adapters.base import AdapterBase
from vlm.configs.base import ConfigBase
from vlm.configs.model.adapter.adapter_config import AdapterClassConfig
from vlm.configs.model.trainable_params.trainable_params_config import TrainableParamsConfig
from vlm.configs.model.vision_encoder_config import VisionEncoderClassConfig
from vlm.vision_encoders.base import VisionEncoderBase

class ModelConfig(ConfigBase):
    model_type: str = Field(..., description="Type of VLM model to train.")
    model_id: str | None = Field(None, description="Model ID or path.")

    vision_encoder: VisionEncoderClassConfig | None = Field(
        None,
        description="Should only be None, if you train with embeddings instead of images or have text-only inputs."
    )

    adapter: AdapterClassConfig | None = Field(
        None,
        description="Should only be None, if you have text-only inputs."
    )

    trainable_params: TrainableParamsConfig

    def build_adapter(
            self,
            vision_embed_dim: int,
            vision_embed_size: tuple[int, int, int],
            text_embed_dim: int
    ) -> AdapterBase:
        if self.adapter is None:
            raise ValueError(f"Cannot build adapter with 'adapter=None'")
        return self.adapter.to_instance(vision_embed_dim, vision_embed_size, text_embed_dim)

    def build_vision_encoder(self) -> VisionEncoderBase:
        if self.vision_encoder is None:
            raise ValueError(f"Cannot build vision encoder with 'vision_encoder=None'")
        return self.vision_encoder.to_instance()