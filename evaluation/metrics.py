from abc import ABC, abstractmethod

from evaluation.evaluate_metrics.bleu.bleu import Bleu
from evaluation.evaluate_metrics.meteor.meteor import Meteor
from evaluation.evaluate_metrics.rouge.rouge import Rouge
from utils.helper import get_process_rank, get_random_uid


class BaseScorer(ABC):
    """
    Base class for all scorers to compute NLP metrics within this VLM framework.
    """
    def __init__(self):
        # Child classes must assign a metric instance here
        self.metric = None

    @abstractmethod
    def __call__(self, preds: list[str], gts: list[str], **kwargs) -> list[float]:
        """
        Returns pairwise scores.
        """
        ...


class BLEUScorer(BaseScorer):
    def __init__(self):
        super().__init__()

        self.metric = Bleu(experiment_id=f"bleu_{get_process_rank()}_{get_random_uid()}")

    def __call__(self, preds: list[str], gts: list[str], max_order: int = 4) -> list[float]:
        """
        BLEU calculates its score on a corpus level, hence we have calculate the pairwise scores iteratively.
        """
        scores = []
        for pred, gt in zip(preds, gts):
            score = self.metric.compute(predictions=[pred], references=[gt], max_order=max_order)["bleu"]
            scores.append(score)
        return scores


class ROUGELScorer(BaseScorer):
    def __init__(self):
        super().__init__()

        self.metric = Rouge(experiment_id=f"rougel_{get_process_rank()}_{get_random_uid()}")

    def __call__(self, preds: list[str], gts: list[str], **kwargs) -> list[float]:
        return self.metric.compute(predictions=preds, references=gts, rouge_types=["rougeL"], use_aggregator=False)["rougeL"]


class METEORScorer(BaseScorer):
    def __init__(self):
        super().__init__()

        self.metric = Meteor(experiment_id=f"meteor_{get_process_rank()}_{get_random_uid()}")

    def __call__(self, preds: list[str], gts: list[str], max_order: int = 4) -> list[float]:
        """
        Computes METEOR scores using a modified 'Meteor' class that returns individual scores
        instead of their mean. See 'README_EVALUATE_METRICS.md' for details.
        """
        return self.metric.compute(predictions=preds, references=gts)["meteor"]
