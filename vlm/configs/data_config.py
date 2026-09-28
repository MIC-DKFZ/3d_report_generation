from typing import Literal
from pydantic import Field, model_validator
from pathlib import Path
import vlm

from torch.utils.data import Dataset

from vlm.configs.base import ConfigBase, RecursiveFindClassConfigBase
from vlm.utils.vision_input_type import VisionInputType
from vlm.utils.find_class_by_name import recursive_find_python_class


class EmbeddingTransformClassConfig(RecursiveFindClassConfigBase):
    folder_to_search = "vlm.data.embedding_transforms"


class DataConfig(ConfigBase):
    """
    Dataset configuration.
    """

    dataset_name: Literal["amosmm", "ctrate", "bonn", "merlin", "toothfairy4"] | None = Field(
        None, description="Name of the dataset."
    )
    train_dataset_json_path: str | None = Field(
        None, description="Path to the train dataset JSON file."
    )
    val_dataset_json_path: str | None = Field(
        None, description="Path to the validation dataset JSON file."
    )
    vision_input_path: str | None = Field(
        None, description="Path to the vision input (either embeddings or images)."
    )
    vision_input_is_sharded: bool = Field(
        False,
        description="Whether 'vision_input_path' points to split folders containing tar shards.",
    )
    vision_input_type: VisionInputType | None = Field(None, description="Vision input type.")
    embed_dim: int | None = Field(
        None, description="Embedding dimension of the embeddings if using"
        " embeddings as input (e.g. 864.)"
    )
    embed_size: tuple[int, int, int] | None = Field(
        None, description="Embedding size if using embeddings as input (e.g. 24x24x24)."
    )

    answer_types: str | list = Field(
        ..., description="Which answer types and subtypes to include during training."
    )

    text_only: bool = Field(
        False,
        description="Whether to only input text and omit any vision tokens."
    )
    embedding_constant: float | Literal["random"] | None = Field(
        None, description="Constant value for language-only baseline. Either a float or 'random'."
    )

    n_train_samples: int | None = Field(None, description="Number of training samples to use.")
    n_val_samples: int | None = Field(1, description="Number of validation samples to use.")

    do_resize: bool = Field(
        False, description="Resize images to 192x192x192 cubes (with lung crops)."
    )
    do_transforms: bool = Field(False, description="Enable nnUNetv2 image augmentation.")
    embedding_transforms: list[EmbeddingTransformClassConfig] = Field(
        default_factory=list,
        description="Embedding-space augmentations to apply to precomputed embeddings during training.",
    )
    do_sentence_shuffle: bool = Field(
        True,
        description="Shuffle the Report sentences for CTRATE or Questions for Amos VQA. The order of Impression and Findings is preserved.",
    )
    sentence_shuffle_p: float = Field(0.5, description="Probability of randomly shuffling the order of sentences.")

    do_truncation: bool = Field(False, description="Whether to truncate tokenized text inputs.")
    truncate_after: str | None = Field(
        None, description="Text marker after which tokenized inputs may be truncated."
    )
    truncate_at: int | None = Field(
        None, description="Number of tokens to keep after the truncation marker."
    )


    is_stacked_2d_embedding: bool = Field(False, description="Set to 'True' for vision encoders such as 'Curia' "
                                                             "where the input embeddings are shaped "
                                                             "(n_slices, class_token + n_patch_tokens, embed_dim).")
    reduce_to_n_slices: int | None = Field(None, description="The number of slices to reduce the stacked 2D embeddings to.")

    image_id_suffix: str = Field(
        "",
        description="Add a suffix to the image IDs provided in the dataset JSON. "
        "WARNING: Only works for CTRATE embeddings.",
    )
    retrieval_json_path: str | None = Field(
        None, description="Path to the retrieval JSON file mapping for the respective dataset."
    )

    retrieval_n_samples: int = Field(
        2, description="How many retrieval samples to use from the retrieval JSON file."
    )

    retrieval_use_image_in_prompt: bool = Field(
        False, description="Whether to use the retrieved image in the prompt."
    )

    @model_validator(mode="after")
    def maybe_set_val_dataset_json_path_to_none(self):
        if isinstance(self.val_dataset_json_path, str):
            normalized_path = self.val_dataset_json_path.strip().lower()
            if normalized_path in {"none", "null"}:
                self.val_dataset_json_path = None
        return self

    @model_validator(mode="after")
    def validate_vision_input_modifiers(self):
        if self.text_only and self.embedding_constant is not None:
            raise ValueError(f"'text_only' and 'embedding_constant' cannot be used together.")
        return self

    @model_validator(mode="after")
    def validate_embed_dims_for_embedding_input(self):
        if self.vision_input_type == VisionInputType.EMBEDDING and (self.embed_dim is None or self.embed_size is None):
            raise ValueError("Missing parameters for embedding mode: "
                             "please specify both 'embed_dim' and 'embed_size' when 'vision_input_type' is 'embedding'.")
        return self

    @model_validator(mode="after")
    def validate_sentence_shuffle(self):
        if not self.do_sentence_shuffle:
            self.sentence_shuffle_p = 0.0
        return self

    @model_validator(mode="after")
    def validate_if_stacked_2d_embeddings(self):
        if self.is_stacked_2d_embedding and self.reduce_to_n_slices is None:
            raise ValueError("Missing parameters for stacked 2D embeddings input: "
                             "please specify 'reduce_to_n_slices' when 'is_stacked_2d_embedding=True'.")
        return self

    @model_validator(mode="after")
    def validate_embedding_augmentations(self):
        if self.embedding_transforms and self.vision_input_type != VisionInputType.EMBEDDING:
            raise ValueError("'embedding_transforms' require 'vision_input_type' to be 'embedding'.")

        return self

    dataset_class_name: str | None = Field(
        None, description="Name of the dataset class."
    )

    folder_to_search: str = "vlm.data"

    def get_dataset_class(self) -> Dataset:
        return recursive_find_python_class(
            folder=str(Path(vlm.__path__[0], *self.folder_to_search.split(".")[1:])),
            class_name=self.dataset_class_name,
            current_module=self.folder_to_search
        )
