"""Check evaluation helpers and saved outputs without loading classifier models."""

import json
import warnings
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from sklearn.exceptions import UndefinedMetricWarning

import eval as evaluation
from vlm.evaluation.report_generation import ctrate_radbert, merlin_modernbert
from vlm.evaluation.report_generation.utils import (
    compute_label_metrics,
    load_reference_labels,
)
from vlm.evaluation.report_generation.utils import labels as label_utils
from vlm.evaluation.vqa import eval_vqa
from vlm.registries.evaluation_registry import CLASSIFIER_EVALUATORS


def test_label_alignment_preserves_requested_order_and_repeated_ids(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({"image_id": ["001", "002"], "finding": [0, 1]}).to_csv(path, index=False)

    labels = load_reference_labels(path, ["002", "001", "002"], ["finding"])

    assert labels.to_dict("list") == {"finding": [1, 0, 1]}


@pytest.mark.parametrize(
    "rows,image_ids,label_cols,message",
    [
        ({"image_id": ["a", "a"], "finding": [0, 1]}, ["a"], ["finding"], "duplicate"),
        ({"image_id": ["a"], "finding": [0]}, ["b"], ["finding"], "requested image_id"),
        ({"image_id": ["a"], "finding": [0]}, ["a"], ["other"], "missing label columns"),
        ({"image_id": [""], "finding": [0]}, ["a"], ["finding"], "empty image_id"),
        ({"image_id": ["a"], "finding": [-1]}, ["a"], ["finding"], "binary labels"),
        ({"image_id": ["a"], "finding": [None]}, ["a"], ["finding"], "binary labels"),
    ],
)
def test_label_alignment_rejects_invalid_labels(tmp_path, rows, image_ids, label_cols, message):
    path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_reference_labels(path, image_ids, label_cols)


def test_vqa_handles_incorrect_and_missing_predictions():
    results = eval_vqa(
        ["The answer is A.", "The answer is B.", "No option selected."],
        ["The answer is A.", "The answer is C.", "The answer is D."],
    )

    assert results["match"] == [True, False, False]
    assert results["answer_pred"] == ["A", "B", "No answer found"]


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


def test_label_metrics_can_use_different_columns_for_crg():
    predictions = pd.DataFrame({"a": [1, 1], "b": [0, 0], "extra": [1, 0]})
    references = pd.DataFrame({"a": [1, 0], "b": [0, 1], "extra": [1, 0]})

    metrics = compute_label_metrics(
        predictions,
        references,
        references.columns,
        crg_label_columns=["a", "b"],
        zero_division=0,
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["crg"] == pytest.approx(1 / 3)


def test_label_metrics_preserve_zero_division_warning_policy():
    predictions = pd.DataFrame({"a": [1, 1], "b": [0, 0]})
    references = pd.DataFrame({"a": [1, 0], "b": [0, 1]})

    with pytest.warns(UndefinedMetricWarning):
        warned = compute_label_metrics(predictions, references, ["a", "b"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silent = compute_label_metrics(predictions, references, ["a", "b"], zero_division=0)

    assert not caught
    assert warned == silent


@pytest.mark.parametrize("value,message", [(0, "zero positive"), (1, "only positive")])
def test_label_metrics_reject_undefined_crg(value, message):
    labels = pd.DataFrame({"a": [value, value]})

    with pytest.raises(ValueError, match=message):
        compute_label_metrics(labels, labels, ["a"], zero_division=0)


@pytest.mark.parametrize("module", [ctrate_radbert, merlin_modernbert])
def test_report_labelers_validate_sample_alignment_before_loading_models(module):
    with pytest.raises(ValueError, match="equal lengths"):
        module.evaluate_reports(["prediction"], [], ["image"])


@pytest.mark.parametrize(
    "module, reference_source",
    [
        (module, source)
        for module in (ctrate_radbert, merlin_modernbert)
        for source in ("supplied", "csv", "model", "calculated")
        if source != "model" or module is merlin_modernbert
    ],
)
def test_report_labelers_resolve_reference_labels_and_compute_metrics(
    tmp_path, monkeypatch, module, reference_source
):
    if module is ctrate_radbert:
        label_columns = module.LABEL_COLUMNS
    else:
        label_columns = ["finding_z", "finding_a"]
        monkeypatch.setattr(
            module.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: SimpleNamespace(
                id2label=dict(enumerate(label_columns)), num_labels=len(label_columns)
            ),
        )
    labels = pd.DataFrame({column: [1, 0] for column in label_columns})
    calls = []

    def predict_labels(reports, *paths):
        calls.append(reports)
        return labels.copy()

    monkeypatch.setattr(module, "predict_labels", predict_labels)
    download = Mock(side_effect=EntryNotFoundError("No bundled labels"))
    monkeypatch.setattr(label_utils, "hf_hub_download", download)
    options = {}
    if reference_source == "supplied":
        options["default_gt_df"] = labels.set_axis([10, 20])
        options["gt_labels_path"] = tmp_path / "unused.csv"
    elif reference_source in ("csv", "model"):
        path = tmp_path / "gt_labels.csv"
        fixed = labels.copy()
        fixed.insert(0, "image_id", ["first", "second"])
        fixed.iloc[::-1].to_csv(path, index=False)
        if reference_source == "csv":
            options["gt_labels_path"] = path
        else:
            download.side_effect = None
            download.return_value = str(path)
    else:
        # Neither evaluator should discover files through a dataset-directory variable.
        monkeypatch.setenv("MERLIN_DATASET_JSONS", str(tmp_path))
        (tmp_path / "gt_labels.csv").write_text("unrelated data")

    metrics, predicted_labels, reference_labels = module.evaluate_reports(
        ["prediction 1", "prediction 2"],
        ["reference 1", "reference 2"],
        ["first", "second"],
        **options,
    )

    assert metrics == pytest.approx(
        {key: 1.0 for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "crg"]}
    )
    pd.testing.assert_frame_equal(predicted_labels, labels)
    pd.testing.assert_frame_equal(reference_labels, labels)
    expected_calls = [["prediction 1", "prediction 2"]]
    if reference_source == "calculated":
        expected_calls.insert(0, ["reference 1", "reference 2"])
        assert (tmp_path / "gt_labels.csv").read_text() == "unrelated data"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["gt_labels.csv"]
    assert calls == expected_calls
    if module is merlin_modernbert and reference_source in ("model", "calculated"):
        download.assert_called_once_with(module.MODEL_ID, "gt_labels.csv")
    else:
        download.assert_not_called()


def test_model_access_errors_are_not_treated_as_missing_labels(monkeypatch):
    # The hub error reads only the headers and request of the HTTP response.
    response = SimpleNamespace(headers={}, request=None)
    download = Mock(side_effect=RepositoryNotFoundError("Cannot access model", response=response))
    monkeypatch.setattr(label_utils, "hf_hub_download", download)
    predict = Mock()

    with pytest.raises(RepositoryNotFoundError):
        label_utils.get_reference_labels(
            ["reference"], ["case"], ["finding"], predict,
            model_id="org/model",
        )

    predict.assert_not_called()


@pytest.mark.parametrize("module", [ctrate_radbert, merlin_modernbert])
def test_partial_reference_csv_is_not_extended_or_reclassified(
    tmp_path, monkeypatch, module
):
    columns = module.LABEL_COLUMNS if module is ctrate_radbert else ["finding_z", "finding_a"]
    if module is merlin_modernbert:
        monkeypatch.setattr(module, "load_label_columns", lambda model_id: columns)
    labels = pd.DataFrame({column: [1, 0] for column in columns})
    path = tmp_path / "gt_labels.csv"
    seed = labels.copy()
    seed.insert(0, "image_id", ["cached", "unrelated"])
    seed.to_csv(path, index=False)
    original = path.read_bytes()
    calls = []

    def predict(reports, *args):
        calls.append(reports)
        return labels.copy()

    monkeypatch.setattr(module, "predict_labels", predict)
    with pytest.raises(ValueError, match="missing.*requested image_id"):
        module.evaluate_reports(
            ["prediction 1", "prediction 2"],
            ["old reference", "new reference"],
            ["cached", "new"],
            gt_labels_path=path,
        )
    assert calls == []
    assert path.read_bytes() == original
