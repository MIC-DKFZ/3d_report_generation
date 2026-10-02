"""Dataset-specific report classifier evaluators."""

from vlm.evaluation.report_generation import ctrate_radbert, merlin_modernbert

CLASSIFIER_EVALUATORS = {
    "ctrate": ctrate_radbert.evaluate_reports,
    "merlin": merlin_modernbert.evaluate_reports,
}


def get_classifier_evaluator(dataset_name: str):
    """Return the classifier evaluator, or None when none is registered."""
    return CLASSIFIER_EVALUATORS.get(dataset_name)
