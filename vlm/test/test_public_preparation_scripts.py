"""Validate public preparation scripts without the internal scripts or package."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import blosc2
import numpy as np
import pandas as pd
import pytest

PUBLIC_ROOT = Path(__file__).resolve().parents[1] / "public_scripts"
RAW_REPORT = "  FINDINGS: normal   lungs .\n1. No change .\n2. Follow-up ; advised .  "
FORMATTED = "FINDINGS: normal lungs.\nIMPRESSION:\nNo change.\nFollow-up; advised."


@pytest.fixture
def exported(tmp_path):
    destination = tmp_path / "public"
    shutil.copytree(PUBLIC_ROOT, destination)
    return destination


def run_script(script, *args, cwd):
    return subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": ""},
        text=True,
        capture_output=True,
    )


def read_json(path):
    return json.loads(path.read_text())


def test_public_scripts_have_no_vlm_imports_or_absolute_source_paths():
    for path in PUBLIC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] == "vlm" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or node.module.split(".")[0] != "vlm"
        assert "/home/j385i" not in path.read_text()
    assert (PUBLIC_ROOT / "embeddings/generate_nnfoundation_embeddings.py").stat().st_size == 0


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_ctrate_from_original_csvs_only(exported, tmp_path, version):
    source = tmp_path / "source/radiology_text_reports"
    source.mkdir(parents=True)
    train_rows = [
        {
            "VolumeName": f"{image_id}.nii.gz",
            "Findings_EN": findings,
            "Impressions_EN": "Not given.",
        }
        for image_id, findings in [
            ("train_1_a_1", "Normal   lungs ."),
            ("train_2_a_1", None),
            ("train_3_a_1", "Validation ."),
            ("train_4_a_1", "Excluded ."),
        ]
    ]
    pd.DataFrame(train_rows).to_csv(source / "train_reports.csv", index=False)
    pd.DataFrame(
        [
            {
                "VolumeName": "valid_1_a_1.nii.gz",
                "Findings_EN": "Normal .",
                "Impressions_EN": "  Follow-up ; advised .  ",
            },
            {
                "VolumeName": "valid_2_a_1.nii.gz",
                "Findings_EN": "Excluded.",
                "Impressions_EN": "Not given.",
            },
        ]
    ).to_csv(source / "validation_reports.csv", index=False)
    assets = exported / "datasets/ctrate"
    selection = {
        "splits": {"train": ["train_1", "train_2", "train_4"], "val": ["train_3"]},
        "exclusions": {
            "v1": ["train_4_a_1", "valid_2_a_1"],
            "v2": ["train_4_a_1", "valid_2_a_1"],
        },
    }
    (assets / "selection.json").write_text(json.dumps(selection))
    output = tmp_path / "generated"
    result = run_script(
        assets / f"create_dataset_jsons_{version}.py",
        "--dataset-path",
        source.parent,
        "--output-dir",
        output,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    splits = {
        split: read_json(output / f"ctrate_{version}_{split}.json")
        for split in ["train", "val", "test", "train_val"]
    }
    expected_train = ["train_1_a_1", "train_2_a_1"] if version == "v1" else ["train_1_a_1"]
    assert [row["image_id"] for row in splits["train"]] == expected_train
    assert [row["image_id"] for row in splits["val"]] == ["train_3_a_1"]
    assert [row["image_id"] for row in splits["test"]] == ["valid_1_a_1"]
    if version == "v1":
        assert splits["train"][0]["answer"] == ("Findings: Normal lungs .\nImpression: Not given.")
        assert splits["test"][0]["answer"] == (
            "Findings: Normal .\nImpression: Follow-up ; advised ."
        )
    else:
        assert splits["train"][0]["answer"] == "Findings: Normal lungs."
        assert splits["test"][0]["answer"] == "Findings: Normal.\nImpression: Follow-up; advised."
    assert (
        sorted(splits["train"] + splits["val"], key=lambda row: row["image_id"])
        == splits["train_val"]
    )
    assert all(row["answer"] == row["full_answer"] for rows in splits.values() for row in rows)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_merlin_from_original_spreadsheet_only(exported, tmp_path, version):
    rows = [
        {"study id": image_id, "Split": split, "Findings": RAW_REPORT}
        for image_id, split in [
            ("train_a", "train"),
            ("val_a", "val"),
            ("val_b", "val"),
            ("test_a", "test"),
            ("excluded_train", "train"),
            ("excluded_test", "test"),
            ("duplicate", "train"),
            ("duplicate", "test"),
        ]
    ]
    source = tmp_path / "reports.xlsx"
    pd.DataFrame(rows).to_excel(source, index=False)
    scripts = exported / "datasets/merlin"
    selection = {
        "splits": {"val": ["val_b"]},
        "exclusions": {
            "v1": ["excluded_train", "excluded_test", "duplicate"],
            "v2": ["excluded_train", "excluded_test", "duplicate"],
        },
    }
    (scripts / "selection.json").write_text(json.dumps(selection))
    output = tmp_path / "generated"
    result = run_script(
        scripts / f"create_dataset_jsons_{version}.py",
        "--reports-xlsx",
        source,
        "--output-dir",
        output,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    splits = {
        split: read_json(output / f"merlin_{version}_{split}.json")
        for split in ["train", "val", "test", "train_val"]
    }
    assert [row["image_id"] for row in splits["train"]] == ["train_a", "val_a"]
    assert [row["image_id"] for row in splits["val"]] == ["val_b"]
    assert [row["image_id"] for row in splits["test"]] == ["test_a"]
    assert all(
        row["answer"]
        == row["full_answer"]
        == (
            "normal lungs.\n1. No change.\n2. Follow-up; advised."
            if version == "v1"
            else FORMATTED
        )
        for records in splits.values()
        for row in records
    )
    assert {path.name for path in output.iterdir()} == {
        f"merlin_{version}_{split}.json" for split in ("train", "val", "test", "train_val")
    }


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_merlin_rejects_unlisted_duplicate_studies(exported, tmp_path, version):
    source = tmp_path / "reports.xlsx"
    pd.DataFrame(
        [
            {"study id": "duplicate", "Split": split, "Findings": RAW_REPORT}
            for split in ("train", "test")
        ]
    ).to_excel(source, index=False)
    scripts = exported / "datasets/merlin"
    (scripts / "selection.json").write_text(
        json.dumps({"splits": {"val": ["val"]}, "exclusions": {"v1": [], "v2": []}})
    )
    result = run_script(
        scripts / f"create_dataset_jsons_{version}.py",
        "--reports-xlsx",
        source,
        "--output-dir",
        tmp_path / "generated",
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "Duplicate study IDs remain" in result.stderr
    assert not (tmp_path / "generated").exists()


def load_retrieval_module():
    pytest.importorskip("faiss")
    path = PUBLIC_ROOT / "retrieval/create_retrieval_files.py"
    spec = importlib.util.spec_from_file_location("public_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retrieval_excludes_all_same_patient_cases_and_matches_exact_cosine(tmp_path):
    module = load_retrieval_module()
    train_ids = [f"train_1_a_{i}" for i in range(4)] + [
        "train_2_a_1",
        "train_3_a_1",
        "train_4_a_1",
    ]
    angles = np.array([0, 0.01, 0.02, 0.03, 0.2, 0.4, 0.6])
    vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    folder = tmp_path / "train_fixed"
    folder.mkdir()
    for image_id, vector in zip(train_ids, vectors):
        blosc2.save_array(vector.reshape(1, 2, 1, 1, 1), str(folder / f"{image_id}_embedding.bl2"))
    index = module.build_index(
        train_ids, tmp_path, "ctrate", "_embedding.bl2", module.PoolingMethod.GAP
    )
    mapping = module.retrieve(
        [train_ids[0]],
        "train",
        tmp_path,
        "ctrate",
        "_embedding.bl2",
        module.PoolingMethod.GAP,
        index,
        train_ids,
        top_k=2,
        batch_size=1,
    )
    assert mapping[train_ids[0]]["k_nearest_neighbor"] == ["train_2_a_1", "train_3_a_1"]
    np.testing.assert_allclose(
        mapping[train_ids[0]]["similarity"], vectors[4:6] @ vectors[0], atol=1e-6
    )


def test_retrieval_cli_outputs_combined_mapping_consumable_by_dataset(exported, tmp_path):
    pytest.importorskip("faiss")
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    for image_id, vector in {"A": [1, 0], "B": [0.8, 0.6], "T": [1, 0.1]}.items():
        blosc2.save_array(
            np.array(vector, dtype=np.float32), str(embeddings / f"{image_id}_embedding.bl2")
        )
    train = tmp_path / "train.json"
    test = tmp_path / "test.json"
    train.write_text(json.dumps([{"image_id": "A"}, {"image_id": "B"}]))
    test.write_text(json.dumps([{"image_id": "T"}]))
    output = tmp_path / "neighbors"
    result = run_script(
        exported / "retrieval/create_retrieval_files.py",
        "--dataset",
        "merlin",
        "--emb-root",
        embeddings,
        "--train-json",
        train,
        "--test-json",
        test,
        "--output-dir",
        output,
        "--top-k",
        1,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    mapping = read_json(output / "retrieval_all.json")
    assert set(mapping) == {"A", "B", "T"}
    assert mapping["A"]["k_nearest_neighbor"] == ["B"]
    assert mapping["T"]["k_nearest_neighbor"] == ["A"]
    assert (output / "train_index.faiss").is_file()
    for query, values in mapping.items():
        assert query not in values["k_nearest_neighbor"]
        assert set(values["k_nearest_neighbor"]) <= {"A", "B"}
        assert len(values["k_nearest_neighbor"]) == len(values["similarity"])


def test_missing_retrieval_embedding_fails_instead_of_silently_dropping_query(tmp_path):
    module = load_retrieval_module()
    with pytest.raises(FileNotFoundError):
        module.load_vector(tmp_path / "missing_embedding.bl2", module.PoolingMethod.GAP)


def test_missing_bundled_selection_is_not_silently_ignored(exported, tmp_path):
    source = tmp_path / "source/radiology_text_reports"
    source.mkdir(parents=True)
    for name in ["train_reports.csv", "validation_reports.csv"]:
        pd.DataFrame(
            [
                {
                    "VolumeName": "train_1_a_1.nii.gz",
                    "Findings_EN": "Normal.",
                    "Impressions_EN": "Not given.",
                }
            ]
        ).to_csv(source / name, index=False)
    (exported / "datasets/ctrate/selection.json").unlink()
    output = tmp_path / "generated"
    result = run_script(
        exported / "datasets/ctrate/create_dataset_jsons_v2.py",
        "--dataset-path",
        source.parent,
        "--output-dir",
        output,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "Required source or bundled support file missing" in result.stderr
    assert not output.exists()
