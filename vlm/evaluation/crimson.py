from CRIMSON import CRIMSONScore
from statistics import mean


def eval_crimson_score(preds: list[str], gts: list[str], batch_size: int = 2) -> dict[str, tuple[float, list[float]]]:
    crimson_scorer = CRIMSONScore()
    crimson_scorer.DEFAULT_MAX_NEW_TOKENS = 6144
    results = crimson_scorer.evaluate_batch(predicted_findings_list=preds, reference_findings_list=gts, batch_size=batch_size)
    scores = [result["crimson_score"] if result else None for result in results]
    valid_scores = [s for s in scores if s is not None]

    return {
        "crimson_score": (mean(valid_scores) if valid_scores else None, scores)
    }