from typing import Literal

from pydantic import Field, model_validator
from torch.utils.data import Dataset

from vlm.configs.base import ConfigBase, RecursiveFindClassConfigBase
from vlm.configs.paths import ConfigPath
from vlm.data.preprocessing.encoder_input import (
    NORMALIZATIONS,
    OUTLIER_METHODS,
    PROCESSING_MODES,
)
from vlm.data.types import VisionInputType


class EmbeddingTransformClassConfig(RecursiveFindClassConfigBase):
    folder_to_search = "vlm.data.embedding_transforms"


def validate_orientation(orientation: str) -> str:
    """The upper-cased orientation code, e.g. 'LPS', naming one of L/R, A/P and S/I each."""
    code = orientation.upper()
    if len(code) != 3 or any(len(set(code) & set(pair)) != 1 for pair in ("LR", "AP", "SI")):
        raise ValueError(
            f"'vision_input_orientation' must name one of L/R, A/P and S/I each, e.g. 'LPS', "
            f"not {code!r}."
        )
    return code


class VisionInputPreprocessingConfig(ConfigBase):
    """
    How the vision inputs (images or the volumes of the embeddings) were made from the CT
    volumes, as the arguments of 'preprocess_encoder_input'
    (vlm/data/preprocessing/encoder_input.py), so that new volumes can be preprocessed the same
    way at inference ('inference_nifti.py'). The encoder input shape is the vision encoder's.
    """

    target_spacing: tuple[float, float, float] | None = Field(
        None,
        description="Spacing (z, y, x) in mm the encoder input shape corresponds to; None: "
        "the volumes keep their own spacing. 'resized' and 'center_crop' resample the volumes to "
        "it. 'resized_direct' does not, and only uses it for the minimum ROI size of "
        "'enforce_min_bbox'.",
    )
    normalization: Literal[NORMALIZATIONS] = Field(
        "z_score", description="Intensity normalization."
    )
    processing_mode: Literal[PROCESSING_MODES] = Field(
        description="How the ROI was fit to the encoder input shape: resized, center "
        "cropped/padded, or resized in one step from the volume's own spacing with "
        "anti-aliasing ('resized_direct')."
    )
    roi_mask: str | None = Field(
        None,
        description="The mask the volumes were cropped to, e.g. 'TotalSegmentator lungs'; "
        "None: the whole volume. Masks of new volumes are passed at inference.",
    )
    mask_label_ids: list[int] | None = Field(
        None, description="Mask labels of the ROI; None: all non-zero labels."
    )
    enforce_min_bbox: bool = Field(
        True,
        description="Enlarge the ROI to at least the encoder input shape at 'target_spacing' "
        "(in voxels of the volume's own spacing without one).",
    )
    outlier_method: Literal[OUTLIER_METHODS] = Field(
        "none", description="Removal of small mask components before cropping."
    )
    outlier_min_voxels: int = Field(
        400, description="Smallest mask component kept by 'connected_components'."
    )

    @model_validator(mode="after")
    def validate_mask_settings_need_a_mask(self):
        if self.roi_mask is None and (
            self.mask_label_ids is not None or self.outlier_method != "none"
        ):
            raise ValueError(
                "'mask_label_ids' and 'outlier_method' describe the ROI mask; set 'roi_mask' too."
            )
        return self

    def preprocessing_kwargs(self, input_shape: tuple[int, int, int]) -> dict:
        """The keyword arguments of 'preprocess_encoder_input'."""
        return {
            "target_spacing": self.target_spacing,
            "normalization": self.normalization,
            "input_shape": tuple(input_shape),
            "processing_mode": self.processing_mode,
            "mask_label_ids": self.mask_label_ids,
            "enforce_min_bbox": self.enforce_min_bbox,
            "outlier_method": self.outlier_method,
            "outlier_min_voxels": self.outlier_min_voxels,
        }


class EmbeddingEncoderConfig(ConfigBase):
    """The vision encoder that made precomputed embeddings, to encode new volumes the same way."""

    class_name: str = Field(
        description="Class in 'vlm/models/vision_encoders', e.g. 'ResEncEncoder'."
    )
    checkpoint_path: str = Field(
        description=(
            "Checkpoint file or Hugging Face repository (optionally '<repository>@<revision>') "
            "the embeddings were made with."
        )
    )


class VisionInputSettings(ConfigBase):
    """
    How a dataset's vision inputs were made, saved once per dataset as
    '<vision_input_path>/vision_input_preprocessing.yml' by the scripts that make them.
    """

    vision_input_orientation: str = Field(
        description="Orientation the volumes were reoriented to, e.g. 'LPS'."
    )
    vision_input_preprocessing: VisionInputPreprocessingConfig
    embedding_encoder: EmbeddingEncoderConfig | None = Field(
        None, description="For embeddings: the vision encoder that made them."
    )

    @model_validator(mode="after")
    def validate_vision_input_orientation(self):
        self.vision_input_orientation = validate_orientation(self.vision_input_orientation)
        return self


class DataConfig(ConfigBase):
    """
    Dataset configuration.
    """

    dataset_name: str | None = Field(
        None, description="Name of the dataset."
    )

    train_dataset_json_path: ConfigPath | None = Field(
        None, description="Path to the train dataset JSON file."
    )
    val_dataset_json_path: ConfigPath | None = Field(
        None, description="Path to the validation dataset JSON file."
    )
    test_dataset_json_path: ConfigPath | None = Field(
        None, description="Path to the test dataset JSON file."
    )
    vision_input_path: ConfigPath | None = Field(
        None, description="Path to the vision input (either embeddings or images)."
    )
    vision_input_is_sharded: bool = Field(
        False,
        description="Whether 'vision_input_path' points to split folders containing tar shards.",
    )
    vision_input_type: VisionInputType | None = Field(None, description="Vision input type.")
    vision_input_orientation: str | None = Field(
        None,
        description="Orientation of the volumes the vision input was made from, as the "
        "SimpleITK/DICOM code of the directions the array axes x, y, z point to (e.g. 'LPS': "
        "x to the patient's left, y to posterior, z to superior). Set by hand, since the "
        "embedding files do not record it; new images are reoriented to it at inference.",
    )
    vision_input_preprocessing: VisionInputPreprocessingConfig | None = Field(
        None,
        description="How the vision inputs were preprocessed from the volumes; used to "
        "preprocess new volumes the same way at inference.",
    )
    embedding_encoder: EmbeddingEncoderConfig | None = Field(
        None,
        description="The vision encoder that made the embeddings ('embedding' mode); used to "
        "encode new volumes the same way at inference.",
    )
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

    n_train_samples: int | None = Field(
        None,
        description="Number of training query samples to use. Retrieval uses the full training JSON.",
    )
    n_val_samples: int | None = Field(1, description="Number of validation samples to use.")

    do_transforms: bool = Field(
        False,
        description="Augment the training images with nnU-Net transforms (image mode only). "
        "Image-mode inputs are otherwise used as stored: preprocess them into encoder inputs "
        "of the encoder's 'patch_size' before training.",
    )
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
    retrieval_json_path: ConfigPath | None = Field(
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
        if self.val_dataset_json_path is not None:
            normalized_path = self.val_dataset_json_path.expression.strip().lower()
            if normalized_path in {"none", "null"}:
                self.val_dataset_json_path = None
        return self

    @model_validator(mode="after")
    def validate_embedding_encoder_is_for_embeddings(self):
        if self.embedding_encoder is not None and self.vision_input_type == VisionInputType.IMAGE:
            raise ValueError(
                "'embedding_encoder' is for runs on embeddings; runs on images use "
                "'model.vision_encoder'."
            )
        return self

    @model_validator(mode="after")
    def validate_vision_input_orientation(self):
        if self.vision_input_orientation is not None:
            self.vision_input_orientation = validate_orientation(self.vision_input_orientation)
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
    def validate_dataset_class(self):
        """Validate the class selected by the dataset and retrieval settings."""
        self.get_dataset_class()
        return self

    @model_validator(mode="after")
    def validate_text_only_retrieval(self):
        if self.text_only and self.retrieval_use_image_in_prompt:
            raise ValueError(
                "'text_only=True' cannot be used with image retrieval. "
                "Disable 'retrieval_use_image_in_prompt' to use report-only retrieval."
            )
        return self

    def get_dataset_class(self) -> type[Dataset]:
        """Select the registered class from the current dataset and retrieval settings."""
        from vlm.registries.dataset_registry import get_dataset_class

        return get_dataset_class(self)
