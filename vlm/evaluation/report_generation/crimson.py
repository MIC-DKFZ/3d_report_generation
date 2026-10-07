"""Evaluate report pairs with the optional CRIMSON scorer."""

from statistics import mean


def evaluate_reports(
    preds: list[str], gts: list[str], batch_size: int = 2
) -> dict[str, tuple[float | None, list[float | None]]]:
    from CRIMSON import CRIMSONScore

    crimson_scorer = CRIMSONScore()
    crimson_scorer.DEFAULT_MAX_NEW_TOKENS = 6144
    results = crimson_scorer.evaluate_batch(
        predicted_findings_list=preds, reference_findings_list=gts, batch_size=batch_size
    )
    scores = [result["crimson_score"] if result else None for result in results]
    valid_scores = [s for s in scores if s is not None]

    return {
        "crimson_score": (mean(valid_scores) if valid_scores else None, scores)
    }
