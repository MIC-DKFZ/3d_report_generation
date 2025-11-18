from typing import Literal
from pydantic import Field, model_validator
from vlm.configs.base import ConfigBase
from vlm.utils.vision_input_type import VisionInputType


class DataConfig(ConfigBase):
    """
    Dataset configuration.
    """
    dataset_name: Literal["amosmm", "ctrate"] | None = Field(None, description="Name of the dataset.")
    train_dataset_json_path: str | None = Field(None, description="Path to the train dataset JSON file.")
    val_dataset_json_path: str | None = Field(None, description="Path to the validation dataset JSON file.")
    vision_input_path: str | None = Field(None, description="Path to the vision input (either embeddings or images).")
    vision_input_type: VisionInputType | None = Field(None, description="Vision input type.")

    embed_dim: int | None = Field(None, description="Embedding dimension when using embeddings as input (e.g. 864).")
    embed_size: tuple[int, int, int] | None = Field(
        None, description="Embedding size if using embeddings as input (e.g. 24x24x24)."
    )

    answer_types: str | list = Field(..., description="Which answer types and subtypes to include during training.")

    text_only: bool = Field(
        False,
        description="Whether to only input text and omit any vision tokens."
    )
    embedding_constant: float | Literal["random"] | None = Field(
        None,
        description="Constant value for language-only baseline. Either a float or 'random'."
    )

    n_train_samples: int | None = Field(None, description="Number of training samples to use.")
    n_val_samples: int = Field(200, description="Number of validation samples to use.")

    do_resize: bool = Field(False, description="Resize images to 192x192x192 cubes (with lung crops).")
    do_transforms: bool = Field(False, description="Enable nnUNetv2 image augmentation.")
    do_sentence_shuffle: bool = Field(True, description="Shuffle the Report sentences for CTRATE or Questions for Amos VQA. The order of Impression and Findings is preserved.")

    image_id_suffix: str = Field("", description="Add a suffix to the image IDs provided in the dataset JSON. "
                                                 "WARNING: Only works for CTRATE embeddings.")

    @model_validator(mode="after")
    def validate_vision_input_modifiers(self):
        if self.text_only and self.embedding_constant is not None:
            raise ValueError(f"'text_only' and 'embedding_constant' cannot be used together.")
        return self

    @model_validator(mode="after")
    def validate_embed_dims_for_embedding_input(self):
        if self.vision_input_type == VisionInputType.EMBEDDING and (self.embed_dim is None or self.embed_size is None):
            raise ValueError("Missing parameters for embedding mode: "
                             "please specify both 'embed_dim' and 'embed_size' when vision_input_type='embedding'.")
        return self