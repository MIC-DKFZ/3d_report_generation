"""Evaluate report pairs with the optional RaTEScore scorer."""

import warnings
from statistics import mean

from loguru import logger


def evaluate_reports(
    preds: list[str], gts: list[str], batch_size: int = 128
) -> dict[str, tuple[float, list[float]]]:
    from RaTEScore import RaTEScore

    # only ~4GB of VRAM usage for batch_size=64, pretty fast overall, so no concerns here
    ratescore = RaTEScore(batch_size=batch_size)

    warnings.filterwarnings(
        "ignore",
        category=SyntaxWarning,
        module="pysbd"
    )
    logger.disable("PyRuSH")

    scores = ratescore.compute_score(candidate_list=preds, reference_list=gts)

    return {
        "rate_score": (mean(scores), scores)
    }
