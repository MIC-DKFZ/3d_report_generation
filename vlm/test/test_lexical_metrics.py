"""Aligning each report once gives the METEOR scores of pycocoevalcap's compute_score."""

import random
import shutil

import pytest
from pycocoevalcap.meteor.meteor import Meteor

from vlm.evaluation.report_generation import lexical_metrics

WORDS = (
    "no evidence of pleural effusion consolidation nodule mass lymphadenopathy in the right "
    "left upper lower lobe heart size is normal mild emphysema small hiatal hernia noted"
).split()


def reports(count, seed):
    rng = random.Random(seed)
    return [" ".join(rng.choices(WORDS, k=rng.randint(8, 40))) for _ in range(count)]


@pytest.mark.skipif(shutil.which("java") is None, reason="METEOR needs Java")
def test_meteor_matches_compute_score_and_keeps_the_statistics():
    preds, gts = reports(30, seed=0), reports(30, seed=1)
    reference = Meteor()
    expected_score, expected_scores = reference.compute_score(
        gts={i: [gt] for i, gt in enumerate(gts)}, res={i: [pred] for i, pred in enumerate(preds)}
    )
    expected_stats = [reference._stat(pred, [gt]) for pred, gt in zip(preds, gts)]

    aggregate, per_case = lexical_metrics.evaluate_reports(preds, gts)
    assert aggregate["meteor_score"] == expected_score
    assert per_case["meteor_score"] == expected_scores
    assert per_case["meteor_stats"] == expected_stats
