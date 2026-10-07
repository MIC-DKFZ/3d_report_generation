"""Evaluate generated reports with BLEU, METEOR, ROUGE-L, and CIDEr."""

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

from vlm.utils import logger


def evaluate_reports(preds: list, gts: list) -> tuple[dict, dict]:
    """
    Returns (aggregate_results, per_case_results).
    per_case_results maps metric names to lists of per-sample scores (same order as preds/gts).
    Failures are indicated by -1.0 in per_case_results and -1.0 in aggregate_results.
    """
    assert len(preds) == len(gts)
    if not preds:
        raise ValueError("RG EVALUATION: received empty lists")

    n = len(preds)
    aggregate = {}
    per_case = {}

    bleu_scorer = Bleu(4)
    meteor_scorer = Meteor()
    rougel_scorer = Rouge()
    cider_scorer = Cider()

    # Put preds and gts into compatible format for Pycocoeval
    in_res = {i: [pred] for i, pred in enumerate(preds)}
    in_gts = {i: [gt] for i, gt in enumerate(gts)}

    # === BLEU ===
    try:
        (bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4), bleu_per_image = (
            bleu_scorer.compute_score(gts=in_gts, res=in_res, verbose=0)
        )
        # bleu_per_image: list of 4 arrays, one per n-gram order
        for order, (agg, per) in enumerate(
            zip(
                [bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4],
                bleu_per_image,
            ),
            start=1,
        ):
            aggregate[f"bleu_score_{order}"] = agg
            per_case[f"bleu_score_{order}"] = list(per)
    except Exception as e:
        logger.error(f"BLEU score calculation failed: {e}")
        for order in range(1, 5):
            aggregate[f"bleu_score_{order}"] = -1.0
            per_case[f"bleu_score_{order}"] = [-1.0] * n

    # === METEOR ===
    try:
        meteor_score, meteor_per_image = meteor_scorer.compute_score(gts=in_gts, res=in_res)
        aggregate["meteor_score"] = meteor_score
        per_case["meteor_score"] = list(meteor_per_image)
        # Collect raw per-sentence statistics strings so calc_statistics.py can
        # bootstrap the *corpus-level* METEOR by resampling stats and re-running EVAL,
        # rather than resampling per-sentence scores (which gives CI for a different quantity).
        meteor_scorer.lock.acquire()
        meteor_stats = [
            meteor_scorer._stat(in_res[i][0], in_gts[i])
            for i in sorted(in_gts.keys())
        ]
        meteor_scorer.lock.release()
        per_case["meteor_stats"] = meteor_stats
    except Exception as e:
        logger.error(f"METEOR score calculation failed: {e}")
        meteor_scorer.lock.release()
        aggregate["meteor_score"] = -1.0
        per_case["meteor_score"] = [-1.0] * n
        per_case["meteor_stats"] = [""] * n

    # === ROUGE ===
    try:
        rougel_score, rougel_per_image = rougel_scorer.compute_score(gts=in_gts, res=in_res)
        aggregate["rougel_score"] = rougel_score.item()
        per_case["rougel_score"] = list(rougel_per_image)
    except Exception as e:
        logger.error(f"ROUGE-L score calculation failed: {e}")
        aggregate["rougel_score"] = -1.0
        per_case["rougel_score"] = [-1.0] * n

    # === CIDEr ===
    try:
        cider_score, cider_per_image = cider_scorer.compute_score(gts=in_gts, res=in_res)
        aggregate["cider_score"] = cider_score.item()
        per_case["cider_score"] = list(cider_per_image)
    except Exception as e:
        logger.error(f"CIDEr score calculation failed: {e}")
        aggregate["cider_score"] = -1.0
        per_case["cider_score"] = [-1.0] * n

    return aggregate, per_case
