import argparse
import random
import numpy as np
import torch
import os
from pydantic import BaseModel
from typing import TypeVar, Any
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


def load_config_file(config_path: str) -> dict:
    file_ext = os.path.splitext(config_path)[1].lower()
    if file_ext == ".yaml" or file_ext == ".yml":
        config_file = load_yaml(config_path)
    elif file_ext == ".json":
        config_file = load_json(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {file_ext}")

    return config_file


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


def is_main_process():
    """
    Check if this is the main process (rank 0).
    As recommended in the JSC tutorial, only rank 0 should handle logging, saving, etc.
    """
    return int(os.environ.get("RANK", "0")) == 0
