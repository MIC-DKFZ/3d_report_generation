from pydantic import Field, model_validator
from pathlib import Path
import vlm

from torch.utils.data import Dataset

from vlm.configs.base import ConfigBase, RecursiveFindClassConfigBase
from vlm.data.types import VisionInputType
from vlm.utils.classes import recursive_find_python_class


class EmbeddingTransformClassConfig(RecursiveFindClassConfigBase):
    folder_to_search = "vlm.data.embedding_transforms"


class DataConfig(ConfigBase):
    """
    Dataset configuration.
    """

    dataset_name: str | None = Field(
        None, description="Name of the dataset."
    )
    dataset_class_name: str | None = Field(
        None, description="Dataset class name; inferred from dataset_name and retrieval settings if omitted."
    )

    train_dataset_json_path: str | None = Field(
        None, description="Path to the train dataset JSON file."
    )
    val_dataset_json_path: str | None = Field(
        None, description="Path to the validation dataset JSON file."
    )
    test_dataset_json_path: str | None = Field(
        None, description="Path to the test dataset JSON file."
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
    sentence_shuffle_p: float = Field(
        0.0, ge=0, le=1,
        description="Probability of shuffling report sentences or VQA options during training. "
        "'0' disables shuffling. Findings and Impression sections remain in order.",
    )

    max_response_tokens: int | None = Field(
        None,
        gt=0,
        description="Maximum number of response tokens, excluding the prompt. 'None' disables truncation.",
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
    def validate_embed_dims_for_embedding_input(self):
        if self.vision_input_type == VisionInputType.EMBEDDING and (self.embed_dim is None or self.embed_size is None):
            raise ValueError("Missing parameters for embedding mode: "
                             "please specify both 'embed_dim' and 'embed_size' when 'vision_input_type' is 'embedding'.")
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

    @model_validator(mode="after")
    def resolve_dataset_class_name(self):
        """Fill a missing class name from the dataset registry."""
        if self.dataset_class_name:
            return self

        from vlm.registries.dataset_registry import get_dataset_class

        self.dataset_class_name = get_dataset_class(self).__name__
        return self

    @model_validator(mode="after")
    def validate_text_only_retrieval(self):
        if self.text_only:
            from vlm.data.datasets import CTRATEDatasetRetrievalWithImage

            if self.retrieval_use_image_in_prompt or issubclass(
                self.get_dataset_class(), CTRATEDatasetRetrievalWithImage
            ):
                raise ValueError(
                    "'text_only=True' cannot be used with image retrieval. "
                    "Disable 'retrieval_use_image_in_prompt' and use a report-only dataset class."
                )
        return self

    def get_dataset_class(self) -> type[Dataset]:
        """Look up the configured dataset class by name."""
        from vlm.data import datasets

        dataset_class = getattr(datasets, self.dataset_class_name, None)
        if dataset_class is not None:
            return dataset_class

        dataset_class = recursive_find_python_class(
            folder=str(Path(vlm.__path__[0], "data")),
            class_name=self.dataset_class_name,
            current_module="vlm.data",
        )
        if dataset_class is None:
            raise ValueError(f"Unknown dataset class: {self.dataset_class_name}")
        return dataset_class
