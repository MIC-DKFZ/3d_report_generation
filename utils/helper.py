import argparse
import random
import numpy as np
import torch
import yaml
import json
import os
from pydantic import BaseModel


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


def load_config_file(config_path: str) -> dict:
    file_ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r") as file:
        if file_ext == ".yaml" or file_ext == ".yml":
            config_file = yaml.safe_load(file)
        elif file_ext == ".json":
            config_file = json.load(file)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")

    return config_file


def merge_config_with_cli_args(default_model: BaseModel) -> BaseModel:
    """
    Overwrites fields of the BaseModel by any matching CLI arguments and
    returns a new BaseModel object.

    Args:
        default_model (BaseModel): A Pydantic BaseModel containing the default config
            that will be overwritten by any matching CLI arguments.

    Returns:
        BaseModel: Returns the merged BaseModel.
    """
    parser = argparse.ArgumentParser()
    for key, value in default_model.model_dump().items():
        arg_type = type(value)
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
                type=arg_type,
                default=value,
                help=f"Set {key} (default: {value})",
            )

    args_parsed, _ = parser.parse_known_args()
    return default_model.model_copy(update=vars(args_parsed))
