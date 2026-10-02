"""Load aligned reference finding labels and calculate label agreement metrics."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def load_reference_labels(
    gt_labels_path: Path,
    image_ids: list[str],
    label_cols: list[str],
) -> pd.DataFrame:
    image_ids = [str(image_id) for image_id in image_ids]
    gt_labels = pd.read_csv(gt_labels_path, dtype={"image_id": str})
    if "image_id" not in gt_labels.columns:
        raise ValueError(f"{gt_labels_path} is missing the required 'image_id' column")

    duplicate_image_ids = gt_labels.loc[
        gt_labels["image_id"].duplicated(keep=False), "image_id"
    ].unique()
    if len(duplicate_image_ids):
        raise ValueError(
            f"{gt_labels_path} contains duplicate image_id values: "
            f"{duplicate_image_ids[:10].tolist()}"
        )

    missing_label_cols = set(label_cols) - set(gt_labels.columns)
    if missing_label_cols:
        raise ValueError(
            f"{gt_labels_path} is missing label columns: {sorted(missing_label_cols)}"
        )

    gt_labels = gt_labels.set_index("image_id")
    missing_image_ids = sorted(set(image_ids) - set(gt_labels.index))
    if missing_image_ids:
        raise ValueError(
            f"{gt_labels_path} is missing {len(missing_image_ids)} requested image_id(s): "
            f"{missing_image_ids[:10]}"
        )

    return gt_labels.loc[image_ids, label_cols].reset_index(drop=True)


def compute_label_metrics(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    label_columns,
    *,
    crg_label_columns=None,
    zero_division="warn",
) -> dict[str, float]:
    """Calculate mean per-label metrics and CRG over explicitly selected findings."""
    label_columns = list(label_columns)
    crg_label_columns = label_columns if crg_label_columns is None else list(crg_label_columns)
    pred = predictions
    gt = references

    results = {}

    # Calculate sklearn metrics
    precs, recs, f1s, accs, bals = [], [], [], [], []

    for col in label_columns:
        y_true = gt[col]
        y_pred = pred[col]
        p = precision_score(y_true, y_pred, zero_division=zero_division)
        r = recall_score(y_true, y_pred, zero_division=zero_division)
        f = f1_score(y_true, y_pred, zero_division=zero_division)
        a = accuracy_score(y_true, y_pred)
        b = balanced_accuracy_score(y_true, y_pred)

        precs.append(p)
        recs.append(r)
        f1s.append(f)
        accs.append(a)
        bals.append(b)

    results.update(
        {
            "accuracy": sum(accs) / len(accs),
            "balanced_accuracy": sum(bals) / len(bals),
            "precision": sum(precs) / len(precs),
            "recall": sum(recs) / len(recs),
            "f1": sum(f1s) / len(f1s),
        }
    )

    # Calculate CRG score
    TP = ((pred[crg_label_columns] == 1) & (gt[crg_label_columns] == 1)).sum().sum()
    FN = ((pred[crg_label_columns] == 0) & (gt[crg_label_columns] == 1)).sum().sum()
    FP = ((pred[crg_label_columns] == 1) & (gt[crg_label_columns] == 0)).sum().sum()

    num_images = len(pred)
    X = len(crg_label_columns) * num_images
    A = int(gt[crg_label_columns].sum().sum())

    if A == 0:
        raise ValueError("CRG undefined: ground truth contains zero positive labels.")

    r = (X - A) / (2 * A)
    U = (X - A) / 2
    s = r * TP - r * FN - FP

    denom = 2 * U - s

    if U == 0:
        raise ValueError("CRG undefined: ground truth contains only positive labels.")

    if denom == 0:
        raise ValueError("CRG undefined: denominator is zero.")

    results["crg"] = U / denom
    return results
