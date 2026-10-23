"""Check MERLIN model metadata and inference without network access or CUDA."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from vlm.evaluation.report_generation import merlin_modernbert


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


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        {1: "finding"},
        {0: "finding", 2: "other"},
    ],
)
def test_invalid_label_metadata_is_rejected(monkeypatch, mapping):
    monkeypatch.setattr(
        merlin_modernbert.AutoConfig,
        "from_pretrained",
        Mock(return_value=SimpleNamespace(id2label=mapping, num_labels=len(mapping or {}))),
    )

    with pytest.raises(ValueError, match="config.id2label"):
        merlin_modernbert.load_label_columns()


@pytest.mark.parametrize("as_dict", [False, True])
def test_hf_inference_uses_config_labels_and_preserves_report_order(monkeypatch, as_dict):
    labels = ["finding_z", "finding_a", "finding_m"]
    config = SimpleNamespace(num_labels=3, id2label=dict(enumerate(labels)))
    config_loader = Mock(return_value=config)

    class FakeClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask):
            assert input_ids.ndim == attention_mask.ndim == 2
            logits = input_ids[:, :1].float() * torch.tensor([1, -1, 0])
            return SimpleNamespace(logits=logits)

    model_loader = Mock(return_value=FakeClassifier())
    tokenizer = Mock(
        side_effect=lambda report, **kwargs: {
            "input_ids": torch.tensor([[int(report), 1, 2]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }
    )
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(merlin_modernbert.AutoConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(
        merlin_modernbert.AutoModelForSequenceClassification, "from_pretrained", model_loader
    )
    monkeypatch.setattr(merlin_modernbert.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(merlin_modernbert.MerlinBertClassifier, "cuda", lambda self: self)
    monkeypatch.setattr(
        merlin_modernbert,
        "DataLoader",
        lambda dataset, sampler, **kwargs: DataLoader(dataset, sampler=sampler, batch_size=2),
    )
    reports = {2: ["0"], 0: ["1"], 1: ["-1"]} if as_dict else ["1", "-1", "0"]

    predicted = merlin_modernbert.predict_labels(reports)

    pd.testing.assert_frame_equal(
        predicted, pd.DataFrame([[1, 0, 1], [0, 1, 1], [1, 1, 1]], columns=labels)
    )
    for loader in (config_loader, tokenizer_loader, model_loader):
        loader.assert_called_once_with(merlin_modernbert.MODEL_ID)
    assert [call.args[0] for call in tokenizer.call_args_list] == ["1", "-1", "0"]
    assert all(call.kwargs["max_length"] == 1024 for call in tokenizer.call_args_list)


def test_local_model_override_is_forwarded_to_predictions(monkeypatch, tmp_path):
    label_columns = ["finding_b", "finding_a"]
    config = SimpleNamespace(num_labels=2, id2label=dict(enumerate(label_columns)))
    config_loader = Mock(return_value=config)
    monkeypatch.setattr(merlin_modernbert.AutoConfig, "from_pretrained", config_loader)
    labels = pd.DataFrame({column: [1, 0] for column in label_columns})
    predictor = Mock(return_value=labels)
    monkeypatch.setattr(merlin_modernbert, "predict_labels", predictor)

    # An unrelated CSV beside a local model must not be inspected.
    (tmp_path / "gt_labels.csv").write_text("invalid data")
    path = tmp_path / "dataset/gt_labels.csv"
    path.parent.mkdir()
    fixed = labels.copy()
    fixed.insert(0, "image_id", ["1", "2"])
    fixed.to_csv(path, index=False)
    _, _, references = merlin_modernbert.evaluate_reports(
        ["pred 1", "pred 2"],
        ["ref 1", "ref 2"],
        ["1", "2"],
        model_id=tmp_path,
        gt_labels_path=path,
    )

    config_loader.assert_called_once_with(str(tmp_path))
    assert [call.args for call in predictor.call_args_list] == [
        (["pred 1", "pred 2"], str(tmp_path)),
    ]
    pd.testing.assert_frame_equal(references, labels)


@pytest.mark.parametrize("bundled_csv", [False, True])
def test_local_model_uses_bundled_labels_or_classifies_references(
    monkeypatch, tmp_path, bundled_csv
):
    columns = ["finding_b", "finding_a"]
    monkeypatch.setattr(merlin_modernbert, "load_label_columns", lambda model_id: columns)
    labels = pd.DataFrame({"finding_b": [1, 0], "finding_a": [0, 1]})
    if bundled_csv:
        saved = labels.copy()
        saved.insert(0, "image_id", ["1", "2"])
        saved.to_csv(tmp_path / "gt_labels.csv", index=False)
    predictor = Mock(return_value=labels)
    monkeypatch.setattr(merlin_modernbert, "predict_labels", predictor)

    _, _, references = merlin_modernbert.evaluate_reports(
        ["pred 1", "pred 2"], ["ref 1", "ref 2"], ["1", "2"], model_id=tmp_path
    )

    pd.testing.assert_frame_equal(references, labels)
    assert predictor.call_count == (1 if bundled_csv else 2)
    assert predictor.call_args.args == (["pred 1", "pred 2"], str(tmp_path))
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        ["gt_labels.csv"] if bundled_csv else []
    )


def test_explicit_gt_csv_is_read_only_and_survives_report_and_setting_changes(
    monkeypatch, tmp_path
):
    columns = ["finding_b", "finding_a"]
    monkeypatch.setattr(merlin_modernbert, "load_label_columns", lambda model_id: columns)
    monkeypatch.setenv("MERLIN_DATASET_JSONS", str(tmp_path))
    (tmp_path / "gt_labels.csv").write_text("unrelated dataset file")
    path = tmp_path / "explicit_labels.csv"
    pd.DataFrame(
        {"image_id": ["001", "002"], "finding_b": [1, 0], "finding_a": [0, 1]}
    ).to_csv(path, index=False)
    original = path.read_bytes()
    expected = pd.DataFrame({"finding_b": [0, 1, 0], "finding_a": [1, 0, 1]})
    predictor = Mock(return_value=expected)
    monkeypatch.setattr(merlin_modernbert, "predict_labels", predictor)

    for max_length, report in ((1024, "original report"), (12, "reformatted report")):
        monkeypatch.setattr(merlin_modernbert, "MAX_LENGTH", max_length)
        metrics, _, references = merlin_modernbert.evaluate_reports(
            ["prediction"] * 3, [report] * 3, ["002", "001", "002"], gt_labels_path=path
        )
        pd.testing.assert_frame_equal(references, expected)
        assert metrics["f1"] == pytest.approx(1.0)

    assert predictor.call_count == 2
    assert all(call.args[0] == ["prediction"] * 3 for call in predictor.call_args_list)
    assert path.read_bytes() == original
    assert sorted(item.name for item in tmp_path.iterdir()) == ["explicit_labels.csv", "gt_labels.csv"]


@pytest.mark.parametrize("source", ["missing_file", "missing_case", "nonbinary"])
def test_missing_or_invalid_fixed_gt_fails_before_prediction(monkeypatch, tmp_path, source):
    monkeypatch.setattr(merlin_modernbert, "load_label_columns", lambda model_id: ["finding"])
    monkeypatch.delenv("MERLIN_DATASET_JSONS", raising=False)
    predictor = Mock()
    monkeypatch.setattr(merlin_modernbert, "predict_labels", predictor)
    if source in ("missing_case", "nonbinary"):
        pd.DataFrame(
            {"image_id": ["other" if source == "missing_case" else "case"], "finding": [-1]}
        ).to_csv(tmp_path / "gt_labels.csv", index=False)

    with pytest.raises((ValueError, FileNotFoundError)):
        merlin_modernbert.evaluate_reports(
            ["prediction"], ["reference"], ["case"], gt_labels_path=tmp_path / "gt_labels.csv"
        )

    predictor.assert_not_called()


@pytest.mark.parametrize(
    "labels, message",
    [
        ({"finding": [0, 1]}, "equal lengths"),
        ({"other": [1]}, "missing label columns"),
        ({"finding": [-1]}, "binary labels"),
    ],
)
def test_supplied_fixed_gt_is_validated_before_prediction(monkeypatch, labels, message):
    monkeypatch.setattr(merlin_modernbert, "load_label_columns", lambda model_id: ["finding"])
    predictor = Mock()
    monkeypatch.setattr(merlin_modernbert, "predict_labels", predictor)

    with pytest.raises(ValueError, match=message):
        merlin_modernbert.evaluate_reports(
            ["prediction"], ["reference"], ["case"], default_gt_df=pd.DataFrame(labels)
        )

    predictor.assert_not_called()
