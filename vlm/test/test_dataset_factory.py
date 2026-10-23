"""Check shared dataset selection without loading images or dataset files."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vlm.configs.data_config import DataConfig
from vlm.data import datasets
from vlm.data.factory import create_dataset
from vlm.registries.dataset_registry import DATASET_REGISTRY, get_dataset_class


@pytest.mark.parametrize(
    "name,classes",
    [
        ("amosmm", [datasets.AMOSMMDataset] * 3),
        ("toothfairy4", [datasets.ToothFairy4Dataset] * 3),
        (
            "ctrate",
            [
                datasets.CTRATEDataset,
                datasets.CTRATEDatasetRetrieval,
                datasets.CTRATEDatasetRetrievalWithImage,
            ],
        ),
        (
            "merlin",
            [
                datasets.MerlinDataset,
                datasets.MerlinDatasetRetrieval,
                datasets.MerlinDatasetRetrievalWithImage,
            ],
        ),
        (
            "bonn",
            [
                datasets.BonnDataset,
                datasets.BonnDatasetRetrieval,
                datasets.BonnDatasetRetrievalWithImage,
            ],
        ),
    ],
)
def test_dataset_variants(name, classes):
    for retrieval_path, with_image, expected in zip(
        [None, "retrieval.json", "retrieval.json"], [True, False, True], classes
    ):
        data = DataConfig(
            dataset_name=name,
            answer_types="report",
            retrieval_json_path=retrieval_path,
            retrieval_use_image_in_prompt=with_image,
        )
        assert get_dataset_class(data) is expected
        assert "dataset_class_name" not in data.model_dump()
        assert data.get_dataset_class() is expected


@pytest.mark.parametrize("split,n_samples", [("train", 10), ("val", 2), ("test", None)])
@pytest.mark.parametrize("patch_size", [None, (16, 16, 16)])
def test_new_registry_entry_and_constructor_arguments(monkeypatch, split, n_samples, patch_size):
    constructor = Mock()
    constructor.__name__ = "CustomDataset"
    monkeypatch.setattr(datasets, "CustomDataset", constructor, raising=False)
    monkeypatch.setitem(DATASET_REGISTRY, "custom", {"default": constructor})
    data = DataConfig(dataset_name="custom", answer_types="report")
    cfg = SimpleNamespace(
        data=data,
        model=SimpleNamespace(
            vision_encoder=SimpleNamespace(patch_size=patch_size) if patch_size else None
        ),
    )
    assert create_dataset(cfg, split, n_samples) is constructor.return_value
    constructor.assert_called_once_with(
        cfg=cfg, split=split, n_samples=n_samples, vision_encoder_patch_size=patch_size
    )


def test_obsolete_class_override_is_rejected_in_new_configs():
    with pytest.raises(ValueError, match="dataset_class_name"):
        DataConfig(
            dataset_name="ctrate",
            dataset_class_name="CTRATEDatasetRetrievalWithImagePrompt1",
            answer_types="report",
        )


def test_unknown_dataset_has_clear_error():
    with pytest.raises(ValueError, match="Unknown dataset name: missing"):
        get_dataset_class(DataConfig(dataset_name="missing", answer_types="report"))


@pytest.mark.parametrize(
    "dataset_class",
    [
        datasets.CTRATEDatasetRetrieval,
        datasets.CTRATEDatasetRetrievalWithImage,
        datasets.CTRATEDatasetRetrievalWithImagePrompt1,
        datasets.MerlinDatasetRetrieval,
        datasets.MerlinDatasetRetrievalWithImage,
        datasets.BonnDatasetRetrieval,
        datasets.BonnDatasetRetrievalWithImage,
    ],
)
def test_retrieval_constructors_forward_sample_limit(monkeypatch, dataset_class):
    class ReachedBaseConstructor(Exception):
        pass

    base_init = Mock(side_effect=ReachedBaseConstructor)
    monkeypatch.setattr(datasets.MedicalImageDataset, "__init__", base_init)
    with pytest.raises(ReachedBaseConstructor):
        dataset_class(cfg=None, split="val", vision_encoder_patch_size=None, n_samples=3)
    base_init.assert_called_once_with(
        cfg=None, split="val", vision_encoder_patch_size=None, n_samples=3
    )
