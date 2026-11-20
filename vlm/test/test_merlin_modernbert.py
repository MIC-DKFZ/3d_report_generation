"""Check MERLIN model metadata and inference without network access or CUDA."""

import json

import pytest

from vlm.evaluation.report_generation import merlin_modernbert


@pytest.fixture(autouse=True)
def fresh_classifier():
    # Each test loads its own fake classifier.
    merlin_modernbert._load_classifier.cache_clear()


def test_labels_come_from_config_in_numeric_output_order(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "modernbert",
                "id2label": {"2": "finding_m", "0": "finding_z", "1": "finding_a"},
                "label2id": {"finding_m": 2, "finding_z": 0, "finding_a": 1},
            }
        )
    )
    # Legacy metadata must not override the model config.
    (tmp_path / "label_cols.json").write_text('["incorrect"]')

    expected_labels = ["finding_z", "finding_a", "finding_m"]
    assert merlin_modernbert.load_label_columns(tmp_path) == expected_labels
