from typing import Literal
from pydantic import Field
from configs.base import ConfigBase


class VisionEncoderConfig(ConfigBase):
    type: Literal["evamae"]
    vision_encoder_path: str | None = Field(None, description="Path to the vision encoder checkpoint.")
    patch_size: tuple[int, int, int] | None = Field(
        None, description="Spatial dimension of the vision encoder input patch size (e.g. 192 x 192 x 192)."
    )
    downsample_factor: tuple[int, int, int] | None = Field(
        None,
        description=(
            "Total spatial downsampling factor (H, W, D) of the vision encoder. "
            "In CNNs, equals the cumulative stride ('output stride'); in ViTs, "
            "corresponds to the patch size used for tokenization."
        ),
    )
    embed_dim: int | None = Field(
        None, description="Embedding dimension of the vision encoder output (e.g. 864)."
    )
