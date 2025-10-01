from typing import Literal, Annotated, Union
from pydantic import Field
from configs.base import ConfigBase
from configs.model.adapter_configs import QwenAdapterConfig
from configs.model.trainable_params_config import TrainableParamsConfig
from configs.model.vision_encoder_config import VisionEncoderConfig


class ModelConfig(ConfigBase):
    model_type: Literal["owl", "qwen"] = Field(..., description="Type of VLM model to train.")
    model_id: str | None = Field(None, description="Model ID or path.")

    vision_encoder: VisionEncoderConfig | None = Field(None, description="Should only be None, if you train with embeddings instead of images.")
    adapter: Annotated[Union[QwenAdapterConfig], Field(discriminator="type")]

    trainable_params: TrainableParamsConfig