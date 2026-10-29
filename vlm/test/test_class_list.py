"""Class list blocks: shuffling, report-only scoring, and class list metrics."""

import json
import random
import re
from argparse import Namespace

import pandas as pd
import pytest

import eval as evaluation
from vlm.data.datasets.ctrate import CTRATEDataset
from vlm.evaluation.report_generation import ctrate_radbert
from vlm.evaluation.report_generation.class_list import (
    evaluate_class_lists,
    match_reference_columns,
)
from vlm.registries.evaluation_registry import CLASSIFIER_EVALUATORS
from vlm.utils.class_list import parse_class_list, split_class_list, strip_class_list

REPORT = (
    "Findings: The heart is normal. Clinic and lab. verification is recommended. "
    "A nodule is seen. No effusion.\nImpression: Lung nodule. Atelectasis."
)
CLASS_LIST = "\n<answer>Lung nodule, Atelectasis</answer>"


def test_split_class_list_only_takes_a_trailing_block():
    assert split_class_list(REPORT + CLASS_LIST) == (REPORT, CLASS_LIST)
    assert split_class_list(REPORT) == (REPORT, "")
    assert split_class_list("<answer>a</answer> then text") == ("<answer>a</answer> then text", "")


def test_strip_and_parse_class_list():
    assert strip_class_list(REPORT + CLASS_LIST) == REPORT
    names = ("Lung nodule", "Atelectasis", "Cardiomegaly")
    assert parse_class_list(REPORT + "\n<answer>atelectasis, Other finding</answer>", names) == [0, 1, 0]
    assert parse_class_list(REPORT + "\n<answer>no findings</answer>", names) == [0, 0, 0]
    assert parse_class_list(REPORT, names) == [0, 0, 0]


def test_parse_class_list_matches_names_inside_entries():
    names = ("Lung nodule", "Pleural effusion", "Atelectasis", "Emphysema")
    text = "<answer>calcified lung-nodules, Bilateral pleural_effusion, Atelectatic band</answer>"
    assert parse_class_list(text, names) == [1, 1, 0, 0]


def test_ctrate_class_names_match_radbert_columns():
    assert match_reference_columns(CTRATEDataset.class_list_names, ctrate_radbert.LABEL_COLUMNS) == list(
        ctrate_radbert.LABEL_COLUMNS
    )


def test_reference_columns_match_after_normalization():
    assert match_reference_columns(["Lung nodule"], ["lung_nodule", "atelectasis"]) == ["lung_nodule"]
    with pytest.raises(ValueError, match="exactly one"):
        match_reference_columns(["Lung nodule"], ["atelectasis"])


def make_ctrate_dataset(answer):
    dataset = CTRATEDataset.__new__(CTRATEDataset)
    dataset.samples = [{"image_id": "x", "question": "Q", "full_answer": answer, "system_text": "S"}]
    dataset.sentence_shuffle_p = 1.0
    dataset.text_only = True
    return dataset


def test_shuffling_keeps_the_class_list_last_and_abbreviations_intact():
    random.seed(0)
    for _ in range(20):
        answer = make_ctrate_dataset(REPORT + CLASS_LIST)[0]["full_answer"]
        report, class_list = split_class_list(answer)
        assert class_list == CLASS_LIST
        assert "Clinic and lab. verification is recommended." in report
        assert report.index("Findings:") < report.index("Impression:")
        findings, impression = report.split("Impression:")
        assert sorted(re.split(r"(?<=\.)\s+", impression.strip())) == ["Atelectasis.", "Lung nodule."]
        assert "A nodule is seen." in findings and "No effusion." in findings


def test_class_list_metrics_need_reference_class_lists():
    labels = pd.DataFrame({"Lung nodule": [1, 0]})
    assert evaluate_class_lists(["a", "b"], ["a", "b"], ("Lung nodule",), labels) == ({}, None)
    assert evaluate_class_lists(["a", "b"], ["a" + CLASS_LIST] * 2, (), labels) == ({}, None)


def test_evaluation_scores_reports_without_class_lists_and_class_lists_separately(
    tmp_path, monkeypatch
):
    records = [
        {"image_id": "0", "output_text_clean": "Report one" + CLASS_LIST, "answer": "Reference one" + CLASS_LIST},
        {"image_id": "1", "output_text_clean": "Report two", "answer": "Reference two\n<answer>no findings</answer>"},
    ]
    (tmp_path / "inference_results.json").write_text(json.dumps(records))
    monkeypatch.setattr(evaluation, "setup_logger", lambda path: None)

    def nlp_metrics(preds, gts):
        assert preds == ["Report one", "Report two"]
        assert gts == ["Reference one", "Reference two"]
        return {}, {}

    reference = pd.DataFrame(0, index=range(2), columns=ctrate_radbert.LABEL_COLUMNS)
    reference.loc[0, ["Lung nodule", "Atelectasis"]] = 1

    def classifier_metrics(preds, gts, image_ids, **options):
        assert preds == ["Report one", "Report two"]
        return {"f1": 0.5}, reference.copy(), reference

    monkeypatch.setattr(evaluation.lexical_metrics, "evaluate_reports", nlp_metrics)
    monkeypatch.setitem(CLASSIFIER_EVALUATORS, "ctrate", classifier_metrics)

    evaluation.main(Namespace(inference_dir=tmp_path, dataset_name="ctrate"))

    metrics = json.loads((tmp_path / "VLM_metrics.json").read_text())
    assert metrics["f1"] == 0.5
    assert metrics["class_list_precision"] == pytest.approx(2 / 18)
    assert metrics["class_list_recall"] == pytest.approx(2 / 18)
    class_lists = pd.read_csv(tmp_path / "class_list_predictions.csv", dtype={"image_id": str})
    assert class_lists["pred_Lung nodule"].tolist() == [1, 0]
    assert class_lists["pred_Cardiomegaly"].tolist() == [0, 0]
