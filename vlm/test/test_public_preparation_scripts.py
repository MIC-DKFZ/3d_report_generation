"""Run the public preparation scripts on their own, copied out of the repository."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

PUBLIC_ROOT = Path(__file__).resolve().parents[1] / "scripts"
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


# Embedding needs the vlm encoders and preprocessing, which inference on new volumes also uses.


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
