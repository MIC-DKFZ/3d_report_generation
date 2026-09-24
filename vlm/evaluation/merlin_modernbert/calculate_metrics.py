import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from vlm.evaluation.merlin_modernbert.utils import merlin_label_cols


def calculate_metrics(pred_classifications_csv: pd.DataFrame, gt_classifications_csv: pd.DataFrame):
    """Same metric suite as the CT-RATE RadBERT path, over the 30 Merlin findings."""
    pred = pred_classifications_csv
    gt = gt_classifications_csv

    results = {}
    precs, recs, f1s, accs, bals = [], [], [], [], []

    for col in merlin_label_cols:
        y_true = gt[col]
        y_pred = pred[col]
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        accs.append(accuracy_score(y_true, y_pred))
        bals.append(balanced_accuracy_score(y_true, y_pred))

    results.update(
        {
            "accuracy": sum(accs) / len(accs),
            "balanced_accuracy": sum(bals) / len(bals),
            "precision": sum(precs) / len(precs),
            "recall": sum(recs) / len(recs),
            "f1": sum(f1s) / len(f1s),
        }
    )

    # CRG score (identical formulation to the CT-RATE path).
    TP = ((pred[merlin_label_cols] == 1) & (gt[merlin_label_cols] == 1)).sum().sum()
    FN = ((pred[merlin_label_cols] == 0) & (gt[merlin_label_cols] == 1)).sum().sum()
    FP = ((pred[merlin_label_cols] == 1) & (gt[merlin_label_cols] == 0)).sum().sum()

    num_images = len(pred)
    X = len(merlin_label_cols) * num_images
    A = int(gt[merlin_label_cols].sum().sum())

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
