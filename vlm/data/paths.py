"""Storage directories shared by dataset loading and retrieval generation."""

from pathlib import Path


def resolve_vision_directory(root: Path, dataset_name: str, split: str) -> Path:
    """Resolve the input directory without probing alternate layouts."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")

    if dataset_name == "merlin":
        return root
    if dataset_name in ("ctrate", "bonn"):
        # CT-RATE validation comes from its training set; preserve Bonn's inherited convention.
        return root / ("valid_fixed" if split == "test" else "train_fixed")

    raise ValueError(f"Unsupported dataset layout: {dataset_name}")
