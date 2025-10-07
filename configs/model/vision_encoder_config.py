from typing import Literal
from pydantic import Field
from configs.base import ConfigBase


class VisionEncoderConfig(ConfigBase):
    type: Literal["evamae"]
    vision_encoder_path: str | None = Field(None, description="Path to vision encoder checkpoint.")