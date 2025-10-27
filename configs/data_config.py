from typing import Literal
from pydantic import Field
from configs.base import ConfigBase
from utils.vision_input_type import VisionInputType


class DataConfig(ConfigBase):
    """
    Dataset configuration.
    """
    dataset_name: Literal["amosmm", "ctrate"] | None = Field(None, description="Name of the dataset.")
    train_dataset_json_path: str | None = Field(None, description="Path to the train dataset JSON file.")
    val_dataset_json_path: str | None = Field(None, description="Path to the validation dataset JSON file.")
    vision_input_path: str | None = Field(None, description="Path to the vision input (either embeddings or images).")
    vision_input_type: VisionInputType | None = Field(None, description="Vision input type.")

    answer_types: str | list = Field(..., description="Which answer types and subtypes to include during training.")

    embedding_constant: float | Literal["random"] | None = Field(
        None,
        description="Constant value for language-only baseline. Either a float or 'random'."
    )

    n_train_samples: int | None = Field(None, description="Number of training samples to use.")
    n_val_samples: int = Field(200, description="Number of validation samples to use.")

    do_resize: bool = Field(False, description="Resize images to 192x192x192 cubes (with lung crops).")
    do_transforms: bool = Field(False, description="Enable nnUNetv2 image augmentation.")
    do_sentence_shuffle: bool = Field(True, description="Shuffle the Report sentences for CTRATE or Questions for Amos VQA. The order of Impression and Findings is preserved.")