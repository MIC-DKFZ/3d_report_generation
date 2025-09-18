from typing import Literal
from pydantic import Field
from configs.base import ConfigBase


class ModelConfig(ConfigBase):
    model_type: Literal["owl", "qwen"] = Field(..., description="Type of VLM model to train.")
    model_id: str = Field(..., description="Model ID or path.")
    vision_encoder_path: str | None = Field(None, description="Path to vision encoder checkpoint.")

    train_only_vision_adapter: bool = Field(
        False,
        description="Train only the vision adapter (Qwen only)."
    )
    train_vision_encoder: bool = Field(
        False,
        description="Train the vision encoder instead of using frozen embeddings."
    )
    freeze_vision_encoder: bool = Field(
        False,
        description="Freeze the vision encoder (no LoRA adapter)."
    )
    reduce_block: tuple[int, int, int] = Field((2, 2, 2))


# class ModelConfig(ConfigBase):
#     vision: VisionEncoderConfig
#     adapter: Annotated[Union[QwenAdapterConfig, TestAdapterConfig], Field(discriminator="type")]
#     language: LanguageDecoderConfig