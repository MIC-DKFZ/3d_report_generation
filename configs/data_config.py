from typing import Literal
from pydantic import Field
from configs.base import ConfigBase


class DataConfig(ConfigBase):
    """
    Dataset configuration.
    """
    dataset_name: Literal["amosmm", "ctrate"] = Field(..., description="Name of the dataset.")
    dataset_path: str = Field(..., description="Path to the dataset JSON file directory.")
    embeddings_path: str = Field(..., description="Path to the vision embeddings.")
    vqa_path: str = Field(..., description="Path to generated VQA data.")

    embedding_constant: float | Literal["random"] | None = Field(
        None,
        description="Constant value for language-only baseline. Either a float or 'random'."
    )

    n_train_samples: int | None = Field(None, description="Number of training samples to use.")
    n_val_samples: int = Field(50, description="Number of validation samples to use (default: 50).")
    split_file: str | None = Field(None, description="Path to cross validation split file (JSON).")
    split_idx: int | None = Field(None, description="Index of the split to use (0-4).")

    dataset_cache: str | None = Field(None, description="Path to cache file for the dataset.")
    dataset_cache_val: str | None = Field(None, description="Path to validation cache file for the dataset.")
    use_csv_reports: bool = Field(False, description="Get reports from CSV file instead of YAML files.")

    do_resize: bool = Field(False, description="Resize images to 192x192x192 cubes (with lung crops).")
    do_transforms: bool = Field(False, description="Enable nnUNetv2 image augmentation.")