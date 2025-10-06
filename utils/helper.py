import argparse
import random
from datetime import datetime
from pathlib import Path

import torch.nn as nn
import numpy as np
import torch
import os
from pydantic import BaseModel
from typing import TypeVar, Any, Literal
from transformers.trainer import TRAINER_STATE_NAME
from utils.load_and_save_files import load_yaml, load_json


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


def add_datetime_tag(s: str, sep: str = "__") -> str:
    return datetime.now().strftime(f"{s}{sep}%Y-%m-%d_%H-%M-%S")


ConfigT = TypeVar("ConfigT", bound=BaseModel)
def merge_config_with_cli_args(default_model: ConfigT) -> ConfigT:
    """
    Update a Pydantic BaseModel with matching CLI arguments.

    Args:
        default_model: A Pydantic model (e.g. GeneralConfig, ModelConfig, ...).

    Returns:
        A new instance of the same model type with updated fields.
    """
    parser = argparse.ArgumentParser()
    for key, field in default_model.model_fields.items():
        value = getattr(default_model, key)
        arg_type = field.annotation

        # Pydantic models or dictionaries should not be passed via CLI.
        # Without this check, we would replace the pydantic models by dictionaries.
        if isinstance(arg_type, (dict, BaseModel)):
            continue

        if arg_type == bool:
            # Handle boolean values with store_true/store_false
            if value:
                parser.add_argument(
                    f"--no-{key}",
                    action="store_false",
                    dest=key,
                    help=f"Disable {key} (default: enabled)",
                )
            else:
                parser.add_argument(
                    f"--{key}",
                    action="store_true",
                    help=f"Enable {key} (default: disabled)",
                )
        else:
            parser.add_argument(
                f"--{key}",
                type=str,           # force CLI args to be parsed as strings; Pydantic will handle casting/validation.
                                    # prevents argparse from using type=NoneType when the default value is None
                default=value,
                help=f"Set {key} (default: {value})",
            )

    args_parsed, _ = parser.parse_known_args()
    return default_model.__class__.model_validate(
        {**default_model.model_dump(), **vars(args_parsed)}
    )


def resolve_ckpt_dir_in_run_dir(run_dir: str | Path, mode: Literal["final", "best"]) -> Path:
    """
    Determine the checkpoint directory within a run directory.

    Args:
        run_dir: Path to the run directory containing checkpoint subdirectories.
        mode: Select the 'final' or the 'best' checkpoint within the run directory.

    Returns:
        Path to the resolved checkpoint directory.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    ckpt_dirs = [d for d in run_dir.glob("checkpoint-*") if d.is_dir()]
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {run_dir}")
    ckpt_dirs_sorted = sorted(ckpt_dirs, key=lambda d: int(d.name.split("-")[-1]))

    final_ckpt_dir = ckpt_dirs_sorted[-1]
    if mode == "final":
        ckpt_dir = final_ckpt_dir
    elif mode == "best":
        final_trainer_state = load_json(final_ckpt_dir / TRAINER_STATE_NAME)
        best_ckpt_dir = Path(final_trainer_state["best_model_checkpoint"])
        ckpt_dir = best_ckpt_dir
    else:
        raise ValueError(f"Unknown checkpoint type to load: mode={mode}")

    return ckpt_dir


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


def is_main_process():
    """
    Check if this is the main process (rank 0).
    As recommended in the JSC tutorial, only rank 0 should handle logging, saving, etc.
    """
    return int(os.environ.get("RANK", "0")) == 0
