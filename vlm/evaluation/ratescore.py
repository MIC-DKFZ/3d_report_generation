from RaTEScore import RaTEScore
from statistics import mean
from loguru import logger
import warnings


def eval_rate_score(preds: list[str], gts: list[str], batch_size: int = 128) -> dict[str, tuple[float, tuple[float, list[float]]]]:
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
