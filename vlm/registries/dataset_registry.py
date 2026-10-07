"""Dataset classes and their retrieval variants."""

from vlm.configs.data_config import DataConfig
from vlm.data.datasets import (
    AMOSMMDataset,
    BonnDataset,
    BonnDatasetRetrieval,
    BonnDatasetRetrievalWithImage,
    CTRATEDataset,
    CTRATEDatasetRetrieval,
    CTRATEDatasetRetrievalWithImage,
    MedicalImageDataset,
    MerlinDataset,
    MerlinDatasetRetrieval,
    MerlinDatasetRetrievalWithImage,
    ToothFairy4Dataset,
)

DATASET_REGISTRY = {
    "amosmm": {"default": AMOSMMDataset},
    "ctrate": {
        "default": CTRATEDataset,
        "retrieval": CTRATEDatasetRetrieval,
        "retrieval_with_image": CTRATEDatasetRetrievalWithImage,
    },
    "merlin": {
        "default": MerlinDataset,
        "retrieval": MerlinDatasetRetrieval,
        "retrieval_with_image": MerlinDatasetRetrievalWithImage,
    },
    "bonn": {
        "default": BonnDataset,
        "retrieval": BonnDatasetRetrieval,
        "retrieval_with_image": BonnDatasetRetrievalWithImage,
    },
    "toothfairy4": {"default": ToothFairy4Dataset},
}


def get_dataset_class(data: DataConfig) -> type[MedicalImageDataset]:
    """Select a registered dataset variant from the dataset and retrieval settings."""
    try:
        variants = DATASET_REGISTRY[data.dataset_name]
    except KeyError:
        raise ValueError(f"Unknown dataset name: {data.dataset_name}") from None

    variant = "default"
    if data.retrieval_json_path is not None:
        variant = "retrieval_with_image" if data.retrieval_use_image_in_prompt else "retrieval"
    # Datasets without retrieval variants keep using their default class.
    return variants.get(variant, variants["default"])
