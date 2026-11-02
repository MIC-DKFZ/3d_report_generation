"""Explicit compatibility rules for configurations saved in 'final_config.yml'."""

from copy import deepcopy
from typing import Any


# Dotted paths of obsolete fields that are accepted but ignored in saved configs.
DEPRECATED_FIELDS: set[str] = {
    "trainer.dataset_kwargs",
    "general.init_optimizer_scheduler_from_checkpoint",
    "data.dataset_class_name",
    "data.folder_to_search",
    "eval.num_dataloader_workers",
    "eval.n_samples",
    "data.do_sentence_shuffle",
    "data.embedding_constant",
    "data.do_truncation",
    "data.truncate_after",
    # Image-mode inputs are preprocessed before training.
    "data.do_resize",
}

# Old dotted path -> current dotted path.
REPLACED_FIELDS: dict[str, str] = {
    "eval.calculate_radbert_score": "eval.calculate_classifier_score",
    "eval.test_dataset_json_path": "data.test_dataset_json_path",
    "data.truncate_at": "data.max_response_tokens",
}


def _get_parent(config: dict, path: str, *, create: bool = False) -> dict | None:
    parent = config
    for key in path.split(".")[:-1]:
        parent = parent.setdefault(key, {}) if create else parent.get(key)
        if not isinstance(parent, dict):
            if create:
                raise ValueError(
                    f"Cannot replace saved config field: '{path}' has a non-dict parent."
                )
            return None
    return parent


def normalize_saved_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Copy and migrate known saved fields, leaving unknown fields for normal validation.

    Return the normalized dictionary and warnings for the caller to log. Supplying
    both an old field and its replacement is an error, even if their values match.
    """
    if not isinstance(config, dict):
        raise ValueError("Saved run configuration must be a dictionary.")

    normalized = deepcopy(config)
    messages = []

    for old_path, new_path in REPLACED_FIELDS.items():
        old_parent = _get_parent(normalized, old_path)
        old_key = old_path.split(".")[-1]
        if old_parent is None or old_key not in old_parent:
            continue

        new_parent = _get_parent(normalized, new_path, create=True)
        assert new_parent is not None  # Creating a parent either succeeds or raises.
        new_key = new_path.split(".")[-1]
        if new_key in new_parent:
            raise ValueError(
                f"Saved config contains both '{old_path}' and its replacement '{new_path}'. "
                "Keep only the replacement field."
            )
        new_parent[new_key] = old_parent.pop(old_key)
        messages.append(f"Saved config field '{old_path}' is deprecated; using '{new_path}'.")

    for path in sorted(DEPRECATED_FIELDS):
        parent = _get_parent(normalized, path)
        key = path.split(".")[-1]
        if parent is not None and key in parent:
            parent.pop(key)
            messages.append(
                f"Saved config field '{path}' is deprecated and ignored; it has no effect."
            )

    return normalized, messages
