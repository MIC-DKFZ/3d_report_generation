import argparse
import json
import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Container

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from pydantic import BaseModel

from vlm.utils.load_and_save_files import load_json, load_yaml
from vlm.utils.paths import VLM_OUTPUT_DIR


def seed_everything(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If you are using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU
        # The two lines below might be needed for full reproducibility with CUDA,
        # but can impact performance. Use them if you absolutely need deterministic behavior.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def getattr_nested(obj, attr_path: str) -> Any:
    parts = attr_path.split(".")
    for part in parts:
        obj = getattr(obj, part)
    return obj


def setattr_nested(obj, attr_path: str, value) -> None:
    parts = attr_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def load_config_file(config_path: str | Path) -> dict:
    file_ext = os.path.splitext(str(config_path))[1].lower()
    if file_ext == ".yaml" or file_ext == ".yml":
        config_file = load_yaml(config_path)
    elif file_ext == ".json":
        config_file = load_json(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {file_ext}")

    return config_file


def collect_image_ids_from_dataset_json(dataset_json_path: str | Path) -> set[str]:
    samples = load_json(dataset_json_path)

    image_ids = {
        str(sample["image_id"])
        for sample in samples
    }
    if not image_ids:
        raise ValueError(f"No image_id entries found in dataset JSON: {dataset_json_path}")
    return image_ids


def collect_image_ids_from_config_dataset_jsons(cfg, splits: list[str]) -> set[str]:
    split_to_dataset_json_path = {
        "train": cfg.data.train_dataset_json_path,
        "val": cfg.data.val_dataset_json_path,
        "test": cfg.eval.test_dataset_json_path,
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


def get_hf_hub_cache_dir() -> str:
    if os.getenv("SLURM_JOB_ID") or os.getenv("LSB_JOBID"):
        cache_dir = os.getenv("HF_HUB_CACHE")
        if cache_dir is None:
            raise EnvironmentError("HF_HUB_CACHE must be set when running in a cluster environment.")
        return cache_dir

    return os.path.expanduser("~/.cache/huggingface/hub")


def get_config_from_plans(plans_file_path):
    # see here https://git.dkfz.de/mic/personal/group4/culrich/nnunet_finetuning/-/blob/fine_tuning_rocket/nnunetv2/training/nnUNetTrainer/variants/pretraining/eva_pretrained.py?ref_type=heads#L167

    plans = load_json(plans_file_path)
    pretrained_config = plans['configurations']['3d_fullres']

    config = {
        # "vit_patch_size": pretrained_config['vit_patch_size'],
        "patch_embed_size": pretrained_config['vit_patch_size'],
        "input_shape": pretrained_config['patch_size'],
        "embed_dim": pretrained_config['embed_dim'],
        "encoder_eva_depth": pretrained_config['encoder_eva_depth'],
        "encoder_eva_numheads": pretrained_config['encoder_eva_numheads'],
    }
    return config


def add_datetime_tag(s: str, sep: str = "__") -> str:
    return datetime.now().strftime(f"{s}{sep}%Y-%m-%d_%H-%M-%S")


def unflatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Turn dotted keys into nested dictionaries."""
    nested: dict[str, Any] = {}
    grouped: dict[str, dict[str, Any]] = {}
    for key, value in d.items():
        head, separator, tail = key.partition(".")
        if separator:
            grouped.setdefault(head, {})[tail] = value
        else:
            nested[head] = value

    for head, values in grouped.items():
        nested[head] = unflatten_dict(values)
    return nested


def add_config_cli_args(
    parser: argparse.ArgumentParser,
    default_model: BaseModel,
    parent_key: str = "",
) -> set[str]:
    """Register dotted CLI options and return paths accepting additional class kwargs."""
    from vlm.configs.base import DynamicClassConfigBase

    dynamic_kwargs_paths: set[str] = set()

    def register_fields(model: BaseModel | dict[str, Any], prefix: str) -> None:
        if isinstance(model, BaseModel):
            fields = (
                (key, field.annotation, getattr(model, key))
                for key, field in type(model).model_fields.items()
            )
        else:
            fields = ((key, type(value), value) for key, value in model.items())

        for key, annotation, value in fields:
            path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, BaseModel):
                register_fields(value, path)
                continue
            if key == "class_kwargs" and isinstance(model, DynamicClassConfigBase):
                assert isinstance(value, dict)
                dynamic_kwargs_paths.add(path)
                register_fields(value, path)
                continue

            if annotation == bool:
                if value:
                    parser.add_argument(
                        f"--{prefix}.no-{key}" if prefix else f"--no-{key}",
                        action="store_false",
                        dest=path,
                        default=argparse.SUPPRESS,
                        help=f"Disable {key} (default: enabled)",
                    )
                else:
                    parser.add_argument(
                        f"--{path}",
                        action="store_true",
                        default=argparse.SUPPRESS,
                        help=f"Enable {key} (default: disabled)",
                    )
            else:
                options = (
                    {"nargs": "+"}
                    if isinstance(value, Container) and not isinstance(value, str)
                    else {}
                )
                parser.add_argument(
                    f"--{path}",
                    type=str,  # Pydantic casts and validates the CLI string.
                    default=argparse.SUPPRESS,
                    help=f"Set {key} (default: {value})",
                    **options,
                )

    register_fields(default_model, parent_key)
    return dynamic_kwargs_paths


def _parse_dynamic_cli_value(value: str) -> Any:
    """Accept JSON values for new class kwargs, leaving ordinary strings unchanged."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def add_new_class_kwarg_cli_args(
    parser: argparse.ArgumentParser,
    cli_args: list[str],
    dynamic_kwargs_paths: set[str],
) -> None:
    """Register previously absent keys beneath known dynamic class kwargs paths."""
    _, unknown_args = parser.parse_known_args(cli_args)
    added_options: set[str] = set()
    for arg in unknown_args:
        if not arg.startswith("--"):
            continue
        option = arg.split("=", 1)[0]
        path = option.removeprefix("--")
        if option in added_options:
            continue
        if any(
            path.startswith(f"{prefix}.") and path != f"{prefix}."
            for prefix in dynamic_kwargs_paths
        ):
            parser.add_argument(
                option, type=_parse_dynamic_cli_value, default=argparse.SUPPRESS
            )
            added_options.add(option)


def recursive_dict_merge(d1: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values from d2 override those from d1.
    Nested dictionaries are merged recursively, all other values are replaced.
    """
    merged = dict(d1)
    for key, value in d2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = recursive_dict_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def print_model_parameters(model: nn.Module, file=None, width=120, **kwargs):
    """Prints model parameters and their requires_grad status to stdout and optionally to a file."""

    def _out(text=""):
        print(text)
        if file is not None:
            print(text, file=file)

    print("\n" + "=" * width)
    print(f"{'Model parameters (requires_grad status):':^{width}}")
    print("=" * width + "\n")

    for name, param in model.named_parameters(**kwargs):
        _out(f"{name:100} requires_grad={param.requires_grad}")

    print("\n" + "=" * width)
    print(f"{'End of model parameter listing':^{width}}")
    print("=" * width + "\n")


def log_param_groups(model: nn.Module, optimizer: Optimizer, file_path=None):
    """Prints optimizer param groups to stdout and optionally to a file."""

    # Build a mapping: param -> name
    param_to_name = {p: n for n, p in model.named_parameters()}

    with open(file_path, "w") as f:
        def print_to_file(s):
            print(s, file=f)

        for i, param_group in enumerate(optimizer.param_groups):
            print_to_file(f"\n=== Parameter Group {i} ===")
            # print_to_file(f"Learning rate : {param_group.get('lr')}",)
            print_to_file(f"Weight decay: {param_group['weight_decay']}")
            print_to_file("Parameters:")

            for p in param_group["params"]:
                name = param_to_name.get(p, "<unnamed>")
                print_to_file(f"  - {name:100} shape={str(tuple(p.shape)):25}")


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def find_run_dir_in_output_dir(run_dir: str):
    matches = list(Path(VLM_OUTPUT_DIR).rglob(run_dir))
    if len(matches) > 1:
        raise ValueError(f"Multiple matches found for run_dir '{run_dir}': {matches}")
    elif not matches:
        raise FileNotFoundError(f"No directory named '{run_dir}' found under {VLM_OUTPUT_DIR}")
    return matches[0]


def get_class_from_registry(class_name: str, registry: list[type]) -> type | None:
    candidates = [c for c in registry if c.__name__ == class_name]
    if candidates:
        return candidates[0]
    return None


def find_sublist(lis: list, sub: list):
    m = len(sub)
    for i in range(len(lis) - m + 1):
        if lis[i:i + m] == sub:
            return i
    return -1


def chunkify(lis: list, num_chunks: int):
    """Split list into 'num_chunks' roughly equal-sized chunks."""
    k, m = divmod(len(lis), num_chunks)
    return [lis[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]


def get_random_uid(length: int = 4) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def is_main_process():
    """
    Check if this is the main process (rank 0).
    As recommended in the JSC tutorial, only rank 0 should handle logging, saving, etc.
    """
    return int(os.environ.get("RANK", "0")) == 0


def get_process_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def get_local_process_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))
