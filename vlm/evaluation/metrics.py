from abc import ABC, abstractmethod

from vlm.evaluation.evaluate_metrics.bleu.bleu import Bleu
from vlm.evaluation.evaluate_metrics.meteor.meteor import Meteor
from vlm.evaluation.evaluate_metrics.rouge.rouge import Rouge
from vlm.utils.helper import get_local_process_rank


class BaseScorer(ABC):
    """
    Base class for all scorers to compute NLP metrics within this VLM framework.
    """
    def __init__(self, metric_name: str):
        # Child classes must assign a metric instance here
        self.metric = None
        self.cache_dir = f"/tmp/HF_METRICS_CACHE"
        self.experiment_id = f"{metric_name}_{get_local_process_rank()}"

    @abstractmethod
    def __call__(self, preds: list[str], gts: list[str], **kwargs) -> list[float]:
        """
        Returns pairwise scores.
        """
        ...


class BLEUScorer(BaseScorer):
    def __init__(self):
        super().__init__(metric_name="bleu")

        self.metric = Bleu(cache_dir=self.cache_dir, experiment_id=self.experiment_id)

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
        super().__init__(metric_name="rougel")

        self.metric = Rouge(cache_dir=self.cache_dir, experiment_id=self.experiment_id)

    def __call__(self, preds: list[str], gts: list[str], **kwargs) -> list[float]:
        return self.metric.compute(predictions=preds, references=gts, rouge_types=["rougeL"], use_aggregator=False)["rougeL"]


class METEORScorer(BaseScorer):
    def __init__(self):
        super().__init__(metric_name="meteor")

        self.metric = Meteor(cache_dir=self.cache_dir, experiment_id=self.experiment_id)

    def __call__(self, preds: list[str], gts: list[str], max_order: int = 4) -> list[float]:
        """
        Computes METEOR scores using a modified 'Meteor' class that returns individual scores
        instead of their mean. See 'README_EVALUATE_METRICS.md' for details.
        """
        return self.metric.compute(predictions=preds, references=gts)["meteor"]
