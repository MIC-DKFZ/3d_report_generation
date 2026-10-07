"""Collect image IDs from dataset JSON records."""

from __future__ import annotations

from pathlib import Path

from vlm.utils.load_and_save_files import load_json


def collect_image_ids_from_dataset_json(dataset_json_path: str | Path) -> set[str]:
    samples = load_json(dataset_json_path)

    image_ids = {str(sample["image_id"]) for sample in samples}
    if not image_ids:
        raise ValueError(f"No image_id entries found in dataset JSON: {dataset_json_path}")
    return image_ids


def collect_image_ids_from_config_dataset_jsons(cfg, splits: list[str]) -> set[str]:
    split_to_dataset_json_path = {
        "train": cfg.data.train_dataset_json_path,
        "val": cfg.data.val_dataset_json_path,
        "test": cfg.data.test_dataset_json_path,
    }

    unknown_splits = sorted(set(splits) - set(split_to_dataset_json_path))
    if unknown_splits:
        raise ValueError(f"Unknown dataset JSON split(s): {unknown_splits}")

    image_ids: set[str] = set()
    seen_paths: set[Path] = set()

    for split in splits:
        dataset_json_path = split_to_dataset_json_path[split]
        if dataset_json_path is None:
            continue
        path = Path(dataset_json_path)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        image_ids.update(collect_image_ids_from_dataset_json(path))

    if not image_ids:
        raise ValueError(f"No image IDs found in configured dataset JSON splits: {splits}")
    return image_ids
