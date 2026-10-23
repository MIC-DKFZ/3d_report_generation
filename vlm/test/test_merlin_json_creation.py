import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from vlm.scripts.data.merlin import create_dataset_jsons_v1, create_dataset_jsons_v2

RAW_REPORT = "  FINDINGS: normal   lungs .\n1. No change .\n2. Follow-up ; advised .  "
FORMATTED_REPORT = "FINDINGS: normal lungs.\nIMPRESSION:\nNo change.\nFollow-up; advised."
V1_REPORT = "normal lungs.\n1. No change.\n2. Follow-up; advised."


@pytest.fixture
def reports_workspace(tmp_path):
    reports = pd.DataFrame(
        [
            {"study id": image_id, "Split": split, "Findings": RAW_REPORT}
            for image_id, split in [
                ("train_b", "train"),
                ("train_a", "train"),
                ("val_b", "val"),
                ("val_a", "val"),
                ("test_a", "test"),
                ("duplicate", "train"),
                ("duplicate", "test"),
            ]
        ]
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    reports.to_excel(inputs / "reports_final.xlsx", index=False)
    (tmp_path / "selection.json").write_text(
        json.dumps({"exclusions": {"v1": ["duplicate"], "v2": ["duplicate"]}})
    )
    return tmp_path


def test_generators_write_versioned_outputs_and_remove_cross_split_duplicates(
    reports_workspace, monkeypatch
):
    root = reports_workspace
    for generator, extra_args in (
        (create_dataset_jsons_v2, ["--n_val_samples", "1"]),
        (create_dataset_jsons_v1, ["--n_val_samples", "1"]),
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "generate",
                "--workspace_dir",
                str(root),
                "--reports_xlsx",
                str(root / "inputs/reports_final.xlsx"),
                "--selection_file",
                str(root / "selection.json"),
                *extra_args,
            ],
        )
        generator.main()

    v2 = {
        split: json.loads((root / f"outputs/merlin_v2_{split}.json").read_text())
        for split in ("train", "val", "test", "train_val")
    }
    assert {split: len(rows) for split, rows in v2.items()} == {
        "train": 3,
        "val": 1,
        "test": 1,
        "train_val": 4,
    }
    ids = {split: {r["image_id"] for r in rows} for split, rows in v2.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"] | ids["val"] == ids["train_val"]
    assert ids["train_val"] == {"train_a", "train_b", "val_a", "val_b"}
    assert ids["test"] == {"test_a"}
    assert all(r["answer"] == FORMATTED_REPORT for rows in v2.values() for r in rows)

    output_dir = root / "outputs"
    assert {f.name for f in output_dir.iterdir()} == {
        f"merlin_{version}_{split}.json"
        for version in ("v1", "v2")
        for split in ("train", "val", "test", "train_val")
    }
    v1_train = json.loads((output_dir / "merlin_v1_train_val.json").read_text())
    v1_test = json.loads((output_dir / "merlin_v1_test.json").read_text())
    assert [r["image_id"] for r in v1_train] == ["train_a", "train_b", "val_a", "val_b"]
    assert [r["image_id"] for r in v1_test] == ["test_a"]
    assert all(r["answer"] == r["full_answer"] == V1_REPORT for r in v1_train)
    assert v1_test[0]["answer"] == v1_test[0]["full_answer"] == V1_REPORT
    for split in ("train", "val"):
        rows = json.loads((output_dir / f"merlin_v1_{split}.json").read_text())
        assert [r["image_id"] for r in rows] == [r["image_id"] for r in v2[split]]
        assert all(r["answer"] == r["full_answer"] == V1_REPORT for r in rows)
    assert {r["image_id"] for r in v1_train} == ids["train"] | ids["val"]
    assert not (root / "intermediate").exists()


def test_v1_keeps_inline_header_and_leading_numbering():
    row = {
        "study id": "case",
        "Findings": "  Findings: normal   lungs . impression: 1. Unchanged .\n2. Follow-up ;  ",
    }
    expected = "Findings: normal lungs. impression: 1. Unchanged.\n2. Follow-up;"
    entry = create_dataset_jsons_v1.standardize_entry(row)
    assert entry["answer"] == entry["full_answer"] == expected
    assert create_dataset_jsons_v2.standardize_entry(row)["answer"] != expected


def test_v1_retains_carriage_returns():
    row = {
        "study id": "case",
        "Findings": (
            "  FINDINGS: normal   lungs .\r\nImpression: 1. Unchanged .\r\n2. Follow-up ;  "
        ),
    }
    expected = "normal lungs.\r\nImpression: 1. Unchanged.\r\n2. Follow-up;"
    entry = create_dataset_jsons_v1.standardize_entry(row)
    assert entry["answer"] == entry["full_answer"] == expected


def test_v1_preserves_report_numbers_and_existing_headers():
    row = {
        "study id": "case",
        "Findings": "FINDINGS: Lesion in segment\n3. Stable.\nIMPRESSION:\n1. Follow-up.",
    }
    expected = "Lesion in segment\n3. Stable.\nIMPRESSION:\n1. Follow-up."
    entry = create_dataset_jsons_v1.standardize_entry(row)
    assert entry["answer"] == entry["full_answer"] == expected


@pytest.mark.parametrize("filename", ["create_dataset_jsons_v2.py", "create_dataset_jsons_v1.py"])
def test_direct_execution_from_outside_repository(reports_workspace, filename):
    root = reports_workspace
    script = Path(create_dataset_jsons_v2.__file__).with_name(filename)
    command = [
        sys.executable,
        str(script),
        "--workspace_dir",
        str(root),
        "--reports_xlsx",
        str(root / "inputs/reports_final.xlsx"),
        "--selection_file",
        str(root / "selection.json"),
        "--n_val_samples",
        "1",
    ]
    if filename == "create_dataset_jsons_v1.py":
        command.extend(
            [
                "--output_json",
                str(root / "custom/train.json"),
                "--test_json",
                str(root / "custom/test.json"),
            ]
        )
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if filename == "create_dataset_jsons_v1.py":
        assert len(json.loads((root / "custom/train.json").read_text())) == 4
        assert len(json.loads((root / "custom/test.json").read_text())) == 1
        assert len(json.loads((root / "custom/merlin_v1_train.json").read_text())) == 3
        assert len(json.loads((root / "custom/merlin_v1_val.json").read_text())) == 1
        assert {path.name for path in (root / "custom").iterdir()} == {
            "train.json",
            "test.json",
            "merlin_v1_train.json",
            "merlin_v1_val.json",
        }


def test_v1_rejects_colliding_output_paths(reports_workspace, monkeypatch):
    output = reports_workspace / "same.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--workspace_dir",
            str(reports_workspace),
            "--output_json",
            str(output),
            "--test_json",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="must be different"):
        create_dataset_jsons_v1.main()
    assert not output.exists()
