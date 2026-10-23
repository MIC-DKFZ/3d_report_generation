"""Check retrieval generation against the layouts used by dataset loading."""

import json
import sys

import blosc2
import numpy as np
import pytest

pytest.importorskip("faiss")

from vlm.data.datasets import BonnDataset, CTRATEDataset, MerlinDataset
from vlm.scripts.retrieval import rag


def save_embedding(directory, image_id, vector, suffix="_embedding.bl2"):
    directory.mkdir(parents=True, exist_ok=True)
    array = np.asarray(vector, dtype=np.float32).reshape(2, 1, 1, 1)
    blosc2.save_array(array, str(directory / f"{image_id}{suffix}"), mode="w")


@pytest.mark.parametrize("dataset_name", ["ctrate", "merlin"])
def test_retrieval_generation_uses_dataset_layout(tmp_path, monkeypatch, dataset_name):
    db_root = tmp_path / "database"
    query_root = tmp_path / "queries"
    json_root = tmp_path / "splits"
    json_root.mkdir()
    output = tmp_path / "retrieval"
    train_ids = ["train_1_a_1", "train_2_a_1"]
    val_id = "train_3_a_1"
    test_id = "valid_4_a_1"

    train_directory = db_root / "train_fixed" if dataset_name == "ctrate" else db_root
    save_embedding(train_directory, train_ids[0], [1, 0])
    save_embedding(train_directory, train_ids[1], [0, 1])
    for split, image_id, vector in [("val", val_id, [1, 0]), ("test", test_id, [0, 1])]:
        expected_directory = query_root
        if dataset_name == "ctrate":
            expected_directory /= "train_fixed" if split == "val" else "valid_fixed"
        for name in ("", "train_fixed", "valid_fixed", "val_fixed"):
            directory = query_root / name
            # Wrong directories contain the opposite vector to reveal fallback mistakes.
            save_embedding(
                directory,
                image_id,
                vector if directory == expected_directory else vector[::-1],
            )

    for split, ids in [("train", train_ids), ("val", [val_id]), ("test", [test_id])]:
        (json_root / f"{dataset_name}_{split}.json").write_text(
            json.dumps([{"image_id": image_id} for image_id in ids])
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag.py",
            "--dataset-name",
            dataset_name,
            "--emb-root",
            str(db_root),
            "--query-emb-root",
            str(query_root),
            "--dataset-json-dir",
            str(json_root),
            "--output-dir",
            str(output),
            "--train-json",
            f"{dataset_name}_train.json",
            "--val-json",
            f"{dataset_name}_val.json",
            "--test-json",
            f"{dataset_name}_test.json",
            "--top-k",
            "1",
        ],
    )
    rag.main()

    metadata = json.loads((output / "gap/train_index_metadata_gap.json").read_text())
    assert metadata["train_emb_subdir"] == ("train_fixed" if dataset_name == "ctrate" else ".")
    assert metadata["train_ids"] == train_ids
    for split, image_id, expected_neighbor in [
        ("val", val_id, train_ids[0]),
        ("test", test_id, train_ids[1]),
    ]:
        result = json.loads((output / f"gap/retrieval_{split}_gap.json").read_text())
        assert result[image_id]["k_nearest_neighbor"] == [expected_neighbor]
        assert result[image_id]["similarity"] == pytest.approx([1.0])
    train_result = json.loads((output / "gap/retrieval_train_gap.json").read_text())
    assert train_result[train_ids[0]]["k_nearest_neighbor"] == [train_ids[1]]
    assert train_result[train_ids[1]]["k_nearest_neighbor"] == [train_ids[0]]


@pytest.mark.parametrize("dataset_name", ["ctrate", "merlin"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_missing_query_does_not_use_other_directories(tmp_path, caplog, dataset_name, split):
    expected_directory = tmp_path
    if dataset_name == "ctrate":
        expected_directory /= "valid_fixed" if split == "test" else "train_fixed"
    image_id = "case"
    suffix = "_custom.bl2"
    for name in ("", "train_fixed", "valid_fixed", "val_fixed"):
        directory = tmp_path / name
        if directory != expected_directory:
            save_embedding(directory, image_id, [1, 0], suffix)

    assert (
        rag.find_query_embedding(
            image_id, {}, tmp_path, suffix, dataset_name=dataset_name, split_name=split
        )
        is None
    )
    assert str(expected_directory / f"{image_id}{suffix}") in caplog.text


@pytest.mark.parametrize("dataset_name", ["ctrate", "merlin"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_query_cache_applies_only_to_training(tmp_path, dataset_name, split):
    directory = tmp_path
    if dataset_name == "ctrate":
        directory /= "valid_fixed" if split == "test" else "train_fixed"
    save_embedding(directory, "case", [1, 0])
    cached = np.asarray([0, 1], dtype=np.float32)
    actual = rag.find_query_embedding(
        "case",
        {"case": cached},
        tmp_path,
        "_embedding.bl2",
        dataset_name=dataset_name,
        split_name=split,
    )
    np.testing.assert_array_equal(actual, cached if split == "train" else [1, 0])


@pytest.mark.parametrize("dataset_class", [CTRATEDataset, MerlinDataset, BonnDataset])
@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("input_type", ["image", "embedding"])
def test_dataset_filenames_and_neighbor_paths(tmp_path, dataset_class, split, input_type):
    dataset = dataset_class.__new__(dataset_class)
    dataset.vision_input_path = tmp_path
    dataset.vision_input_type = input_type
    dataset.image_id_suffix = "_custom"
    dataset.split = split
    filename = "case_custom_embedding.bl2" if input_type == "embedding" else "case.b2nd"
    query_directory = neighbor_directory = tmp_path
    if dataset_class is not MerlinDataset:
        query_directory /= "valid_fixed" if split == "test" else "train_fixed"
        neighbor_directory /= "train_fixed"
    assert dataset._get_vision_input_path("case") == query_directory / filename
    assert (
        dataset._get_vision_input_path("case", split_path="train") == neighbor_directory / filename
    )
