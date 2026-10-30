"""Dataset-specific report classifier evaluators."""

from vlm.evaluation.report_generation import ctrate_radbert, merlin_modernbert

CLASSIFIER_EVALUATORS = {
    "ctrate": ctrate_radbert.evaluate_reports,
    "merlin": merlin_modernbert.evaluate_reports,
}

CLASSIFIER_LOADERS = {
    "ctrate": ctrate_radbert.load_classifier,
    "merlin": merlin_modernbert.load_classifier,
}


def get_classifier_evaluator(dataset_name: str):
    """Return the classifier evaluator, or None when none is registered."""
    return CLASSIFIER_EVALUATORS.get(dataset_name)


def load_classifier(dataset_name: str):
    """Load and cache the classifier model that the dataset's evaluator uses."""
    return CLASSIFIER_LOADERS[dataset_name]()
