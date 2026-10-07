"""Construct datasets from a run configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vlm.configs.run_config import RunConfig
    from vlm.data.datasets import MedicalImageDataset


def create_dataset(
    cfg: RunConfig,
    split: Literal["train", "val", "test"],
    n_samples: int | Literal["unique"] | None = None,
) -> MedicalImageDataset:
    """Construct a dataset with the configured class and vision encoder patch size."""
    dataset_class = cfg.data.get_dataset_class()
    patch_size = cfg.model.vision_encoder.patch_size if cfg.model.vision_encoder else None
    return dataset_class(
        cfg=cfg,
        split=split,
        n_samples=n_samples,
        vision_encoder_patch_size=patch_size,
    )
