"""Check evaluation helpers and saved outputs without loading classifier models."""

import json
from argparse import Namespace

import pandas as pd
import pytest

import eval as evaluation
from vlm.evaluation.report_generation.utils import (
    compute_label_metrics,
)
from vlm.registries.evaluation_registry import CLASSIFIER_EVALUATORS


@pytest.mark.parametrize(
    "dataset_name,with_classifier",
    [("ctrate", False), ("ctrate", True), ("merlin", False), ("merlin", True), ("other", False)],
)
def test_evaluation_writes_metrics_and_optional_classifications(
    tmp_path, monkeypatch, dataset_name, with_classifier
):
    predictions = ["First\nreport", "Second report"]
    answers = ["First\nreference", "Second reference"]
    records = [
        {"image_id": str(i), "output_text_clean": pred, "answer": gt}
        for i, (pred, gt) in enumerate(zip(predictions, answers))
    ]
    (tmp_path / "inference_results.json").write_text(json.dumps(records))
    monkeypatch.setattr(evaluation, "setup_logger", lambda path: None)

    def nlp_metrics(preds, gts):
        assert preds == ["First report", "Second report"]
        assert gts == ["First reference", "Second reference"]
        return {"bleu_score_1": 0.5}, {"bleu_score_1": [0.25, 0.75]}

    gt_labels_path = tmp_path / "gt_labels.csv" if dataset_name == "merlin" else None

    def classifier_metrics(preds, gts, image_ids, **options):
        assert image_ids == ["0", "1"]
        assert options == ({"gt_labels_path": gt_labels_path} if gt_labels_path else {})
        if not with_classifier:
            return {}, None, None
        return (
            {"f1": 0.8},
            pd.DataFrame({"finding": [1, 0]}),
            pd.DataFrame({"finding": [1, 1]}),
        )

    monkeypatch.setattr(evaluation.lexical_metrics, "evaluate_reports", nlp_metrics)
    if dataset_name in CLASSIFIER_EVALUATORS:
        monkeypatch.setitem(CLASSIFIER_EVALUATORS, dataset_name, classifier_metrics)

    evaluation.main(
        Namespace(inference_dir=tmp_path, dataset_name=dataset_name, gt_labels_path=gt_labels_path)
    )

    expected_metrics = {"bleu_score_1": 0.5}
    if with_classifier:
        expected_metrics["f1"] = 0.8
    assert json.loads((tmp_path / "VLM_metrics.json").read_text()) == expected_metrics
    per_case = pd.read_csv(tmp_path / "metrics_per_case.csv", dtype={"image_id": str})
    assert per_case["bleu_score_1"].tolist() == [0.25, 0.75]
    assert per_case["image_id"].tolist() == ["0", "1"]
    classification_path = tmp_path / "radbert_predictions.csv"
    assert classification_path.exists() == with_classifier
    if with_classifier:
        labels = pd.read_csv(classification_path)
        assert labels["pred_finding"].tolist() == [1, 0]
        assert labels["gt_finding"].tolist() == [1, 1]


def test_label_metrics_average_each_finding_and_compute_crg():
    predictions = pd.DataFrame({"a": [1, 1], "b": [0, 0]})
    references = pd.DataFrame({"a": [1, 0], "b": [0, 1]})

    metrics = compute_label_metrics(predictions, references, ["a", "b"], zero_division=0)

    assert metrics == pytest.approx(
        {
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "precision": 0.25,
            "recall": 0.5,
            "f1": 1 / 3,
            "crg": 1 / 3,
        }
    )
