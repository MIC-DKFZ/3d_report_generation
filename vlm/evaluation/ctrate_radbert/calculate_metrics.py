import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, accuracy_score

from vlm.evaluation.ctrate_radbert.utils import ctrate_label_cols


def calculate_metrics(pred_classifications_csv: pd.DataFrame, gt_classifications_csv: pd.DataFrame):
    pred = pred_classifications_csv
    gt = gt_classifications_csv

    results = {}
    
    # Calculate sklearn metrics
    precs, recs, f1s, accs, bals = [], [], [], [], []

    for col in gt.columns:
        y_true = gt[col]
        y_pred = pred[col]
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)
        a = accuracy_score(y_true, y_pred)
        b = balanced_accuracy_score(y_true, y_pred)

        precs.append(p)
        recs.append(r)
        f1s.append(f)
        accs.append(a)
        bals.append(b)

    results.update({
        "accuracy": sum(accs) / len(accs),
        "balanced_accuracy": sum(bals) / len(bals),
        "precision": sum(precs) / len(precs),
        "recall": sum(recs) / len(recs),
        "f1": sum(f1s) / len(f1s),
    })

    # Calculate CRG score
    TP = ((pred[ctrate_label_cols] == 1) & (gt[ctrate_label_cols] == 1)).sum().sum()
    FN = ((pred[ctrate_label_cols] == 0) & (gt[ctrate_label_cols] == 1)).sum().sum()
    FP = ((pred[ctrate_label_cols] == 1) & (gt[ctrate_label_cols] == 0)).sum().sum()

    num_images = len(pred)
    X = len(ctrate_label_cols) * num_images
    A = int(gt[ctrate_label_cols].sum().sum())

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
