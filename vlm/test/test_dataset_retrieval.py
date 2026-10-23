"""Exercise retrieval augmentation through the normal sample-loading path."""

import json
from pathlib import Path
from types import SimpleNamespace

import blosc2
import numpy as np
import pytest

from vlm.configs.data_config import DataConfig
from vlm.data import datasets
from vlm.data.datasets.retrieval import ImageRetrievalMixin, TextRetrievalMixin


@pytest.mark.parametrize(
    "dataset_class,dataset_name",
    [
        (datasets.CTRATEDataset, "ctrate"),
        (datasets.MerlinDataset, "merlin"),
        (datasets.BonnDataset, "bonn"),
        (datasets.CTRATEDatasetRetrieval, "ctrate"),
        (datasets.CTRATEDatasetRetrievalWithImage, "ctrate"),
        (datasets.CTRATEDatasetRetrievalWithImagePrompt1, "ctrate"),
        (datasets.MerlinDatasetRetrieval, "merlin"),
        (datasets.MerlinDatasetRetrievalWithImage, "merlin"),
        (datasets.BonnDatasetRetrieval, "bonn"),
        (datasets.BonnDatasetRetrievalWithImage, "bonn"),
    ],
)
@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("n_neighbors", [0, 2, 5])
@pytest.mark.parametrize("n_samples", [None, 1])
def test_sample_prompts_include_retrieved_context(
    tmp_path, dataset_class, dataset_name, split, n_neighbors, n_samples
):
    image_ids = ["train_1_a_1", "train_2_a_1", "train_3_a_1", "train_4_a_1", "valid_5_a_1"]
    samples = [
        {
            "image_id": image_id,
            "question": "Generate a report.",
            "answer": f"Reference report for {image_id}.",
            "full_answer": f"Target report for {image_id}.",
            "answer_type": "TEXT",
            "answer_subtype": "REPORT",
            "system_text": "Medical assistant.",
        }
        for image_id in image_ids
    ]
    for name, records in [("train", samples[:3]), ("val", samples[3:4]), ("test", samples[4:])]:
        (tmp_path / f"{name}.json").write_text(json.dumps(records))

    neighbors = image_ids[1:3]
    retrieval = {
        image_id: {
            "k_nearest_neighbor": ["missing_case", *neighbors, image_ids[4]],
            "similarity": [0.95, 0.9, 0.8, 0.7],
        }
        for image_id in image_ids
    }
    (tmp_path / "retrieval.json").write_text(json.dumps(retrieval))
    embeddings = {}
    for index, image_id in enumerate(image_ids):
        embedding = np.full((2, 2, 2, 2), index, dtype=np.float32)
        embeddings[image_id] = embedding
        directory = tmp_path
        if dataset_name != "merlin":
            directory /= "valid_fixed" if image_id.startswith("valid_") else "train_fixed"
        directory.mkdir(exist_ok=True)
        blosc2.save_array(embedding, str(directory / f"{image_id}_embedding.bl2"), mode="w")

    uses_retrieval = issubclass(dataset_class, TextRetrievalMixin)
    uses_images = issubclass(dataset_class, ImageRetrievalMixin)
    cfg = SimpleNamespace(
        data=DataConfig(
            dataset_name=dataset_name,
            answer_types="TEXT",
            train_dataset_json_path=str(tmp_path / "train.json"),
            val_dataset_json_path=str(tmp_path / "val.json"),
            test_dataset_json_path=str(tmp_path / "test.json"),
            vision_input_path=str(tmp_path),
            vision_input_type="embedding",
            embed_dim=2,
            embed_size=(2, 2, 2),
            retrieval_json_path=str(tmp_path / "retrieval.json") if uses_retrieval else None,
            retrieval_n_samples=n_neighbors,
            retrieval_use_image_in_prompt=uses_images,
            n_train_samples=n_samples,
        )
    )
    dataset = dataset_class(
        cfg=cfg, vision_encoder_patch_size=None, split=split, n_samples=n_samples
    )
    assert len(dataset) == (1 if n_samples == 1 or split != "train" else 3)
    query_id = dataset.samples[0]["image_id"]
    # The singular path resolver and loader must retain their single-image contracts.
    query_path = dataset._get_vision_input_path(query_id)
    assert isinstance(query_path, Path)
    np.testing.assert_array_equal(dataset._load_vision_input(query_path), embeddings[query_id])
    if uses_retrieval:
        # Limits apply to queries, while every split uses the full training reference pool.
        assert {case["image_id"] for case in dataset.samples_retrieval} == set(image_ids[:3])
        assert dataset.retrieval_dict[query_id] == {
            "k_nearest_neighbor": neighbors,
            "similarity": [0.9, 0.8],
        }
    expected_neighbors = neighbors[:n_neighbors] if uses_retrieval else []

    # Repeated loading must not accumulate context in the stored dataset sample.
    for _ in range(2):
        sample = dataset[0]
        for messages in [sample["formatted_qa"], sample["formatted_qa_prompt_only"]]:
            content = messages[1]["content"]
            reports = [
                item["text"] for item in content if "Reference report" in item.get("text", "")
            ]
            assert len(reports) == len(expected_neighbors)
            for report, image_id in zip(reports, expected_neighbors):
                assert f"Reference report for {image_id}." in report
            vision_items = [item for item in content if item["type"] == "vision"]
            assert len(vision_items) == 1 + (len(expected_neighbors) if uses_images else 0)
            assert [item["vision_input_path"] for item in vision_items[1:]] == (
                expected_neighbors if uses_images else []
            )

        assert sample["formatted_qa"][-1] == {
            "role": "assistant",
            "content": [{"type": "text", "text": dataset.samples[0]["full_answer"]}],
        }
        assert [message["role"] for message in sample["formatted_qa_prompt_only"]] == [
            "system",
            "user",
        ]
        scans = sample["vision_input"] if uses_images else [sample["vision_input"]]
        expected_scan_ids = [sample["image_id"]] + (expected_neighbors if uses_images else [])
        assert len(scans) == len(expected_scan_ids)
        for scan, image_id in zip(scans, expected_scan_ids):
            np.testing.assert_array_equal(scan, embeddings[image_id])


@pytest.mark.parametrize("n_neighbors", [0, 2])
@pytest.mark.parametrize("dataset_name", ["ctrate", "merlin", "bonn"])
def test_text_only_rejects_image_retrieval(dataset_name, n_neighbors):
    with pytest.raises(ValueError, match="'text_only=True' cannot be used with image retrieval"):
        DataConfig(
            dataset_name=dataset_name,
            answer_types="TEXT",
            text_only=True,
            retrieval_json_path="retrieval.json",
            retrieval_use_image_in_prompt=True,
            retrieval_n_samples=n_neighbors,
        )


@pytest.mark.parametrize("dataset_name", ["ctrate", "merlin", "bonn"])
def test_text_only_allows_report_only_retrieval(dataset_name):
    data = DataConfig(
        dataset_name=dataset_name,
        answer_types="TEXT",
        text_only=True,
        retrieval_json_path="retrieval.json",
        retrieval_use_image_in_prompt=False,
    )
    assert data.get_dataset_class() is {
        "ctrate": datasets.CTRATEDatasetRetrieval,
        "merlin": datasets.MerlinDatasetRetrieval,
        "bonn": datasets.BonnDatasetRetrieval,
    }[dataset_name]
