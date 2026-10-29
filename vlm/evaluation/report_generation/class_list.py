"""Score the class lists that models generate after their reports."""

import re

import pandas as pd

from vlm.evaluation.report_generation.utils import compute_label_metrics
from vlm.utils import logger
from vlm.utils.class_list import has_class_list, parse_class_list


def _normalize_class_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def match_reference_columns(class_names, reference_columns) -> list[str]:
    """Map each class name to its reference label column, e.g. 'Lung nodule' to 'lung_nodule'."""
    columns_by_name = {}
    for column in reference_columns:
        columns_by_name.setdefault(_normalize_class_name(column), []).append(column)

    matched = []
    for name in class_names:
        columns = columns_by_name.get(_normalize_class_name(name), [])
        if len(columns) != 1:
            raise ValueError(
                f"Class {name!r} must match exactly one reference label column, found {columns}."
            )
        matched.append(columns[0])
    return matched


def evaluate_class_lists(
    preds: list[str],
    gts: list[str],
    class_names,
    reference_labels: pd.DataFrame | None,
) -> tuple[dict, pd.DataFrame | None]:
    """
    Score generated class lists against the classifier's reference labels.

    Runs only when the dataset defines class names, the references carry class lists, and
    reference labels exist. Returns (metrics, predicted labels with reference column names).
    """
    if not class_names or reference_labels is None or not any(has_class_list(gt) for gt in gts):
        return {}, None

    columns = match_reference_columns(class_names, reference_labels.columns)
    num_missing = sum(not has_class_list(pred) for pred in preds)
    if num_missing:
        logger.warning_main(
            f"{num_missing}/{len(preds)} generated answers contain no class list; "
            "they count as predicting no classes."
        )
    pred_df = pd.DataFrame([parse_class_list(pred, class_names) for pred in preds], columns=columns)
    metrics = compute_label_metrics(
        pred_df, reference_labels, label_columns=columns, zero_division="warn"
    )
    return metrics, pred_df
