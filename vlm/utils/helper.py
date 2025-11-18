import argparse
import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import torch
import torch.nn as nn
import transformers.trainer
from pydantic import BaseModel
from transformers.trainer import TRAINER_STATE_NAME

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


ConfigT = TypeVar("ConfigT", bound=BaseModel)
def merge_config_with_cli_args(default_model: ConfigT, ignore_keys: str | list[str] | None = None, parent_key: str = "") -> ConfigT:
    """
    Update a Pydantic BaseModel with matching CLI arguments.

    Args:
        default_model: A Pydantic model (e.g. GeneralConfig, ModelConfig, ...).
        ignore_keys: A list of keys for values that should not be merged even if provided via CLI.
        parent_key: The subconfiguration that is to be overwritten by the CLI arguments.

    Returns:
        A new instance of the same model type with updated fields.
    """
    if ignore_keys is None:
        ignore_keys = []
    elif isinstance(ignore_keys, str):
        ignore_keys = [ignore_keys]

    if parent_key != "":
        parent_key = parent_key + "."

    parser = argparse.ArgumentParser()
    for key, field in default_model.model_fields.items():

        if key in ignore_keys:
            continue

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
                    f"--{parent_key}no-{key}",
                    action="store_false",
                    dest=key,
                    help=f"Disable {key} (default: enabled)",
                )
            else:
                parser.add_argument(
                    f"--{parent_key}{key}",
                    action="store_true",
                    help=f"Enable {key} (default: disabled)",
                )
        else:
            parser.add_argument(
                f"--{parent_key}{key}",
                type=str,           # force CLI args to be parsed as strings; Pydantic will handle casting/validation.
                                    # prevents argparse from using type=NoneType when the default value is None
                default=value,
                help=f"Set {key} (default: {value})",
            )

    args_parsed, _ = parser.parse_known_args()
    args_parsed = {k.removeprefix(parent_key): v for k, v in vars(args_parsed).items()}
    return default_model.__class__.model_validate(
        {**default_model.model_dump(), **args_parsed}
    )


def resolve_ckpt_dir_in_run_dir(run_dir: str | Path, mode: Literal["final", "best", "best_metric"]) -> Path:
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
        raise FileNotFoundError(f"The specified run directory does not exist: {run_dir}")
    if mode == "best_metric":
        our_best_ckpt_dir = Path(run_dir, "best_model")
        if not our_best_ckpt_dir.is_dir():
            raise FileNotFoundError(f"No 'best_model' directory found in {run_dir}")
        ckpt_dir = our_best_ckpt_dir
    else:
        ckpt_dirs = [d for d in run_dir.glob("checkpoint-*") if d.is_dir()]
        if not ckpt_dirs:
            raise FileNotFoundError(f"No HF trainer checkpoint directories found in {run_dir}")
        ckpt_dirs_sorted = sorted(ckpt_dirs, key=lambda d: int(d.name.split("-")[-1]))

        final_ckpt_dir = ckpt_dirs_sorted[-1]
        if mode == "final":
            ckpt_dir = final_ckpt_dir
        elif mode == "best":
            final_trainer_state = load_json(final_ckpt_dir / TRAINER_STATE_NAME)
            best_ckpt_dir = Path(final_trainer_state["best_model_checkpoint"])
            if not best_ckpt_dir.is_dir():
                raise FileNotFoundError(f"The HF trainer 'best_model_checkpoint' directory does not exist in {run_dir}")
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


def print_param_groups(trainer: transformers.trainer.Trainer, file=None):
    """Prints optmizer param groups to stdout and optionally to a file."""

    def _out(text=""):
        print(text)
        if file is not None:
            print(text, file=file)

    trainer.create_optimizer()
    optimizer = trainer.optimizer

    # Build a mapping: param -> name
    param_to_name = {p: n for n, p in trainer.model.named_parameters()}

    for i, param_group in enumerate(optimizer.param_groups):
        _out(f"\n=== Param group {i} ===")
        _out(f"Learning rate : {param_group.get('lr')}")
        _out(f"Weight decay  : {param_group['weight_decay']}")
        _out("Parameters:")

        for p in param_group["params"]:
            name = param_to_name.get(p, "<unnamed>")
            _out(f"  - {name}   shape={tuple(p.shape)}")


def get_trainable_state_dict(model: nn.Module):
    """Return a state dict containing only parameters with requires_grad=True."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    state_dict = {name: p for name, p in model.state_dict().items() if name in trainable_names}

    return state_dict


def find_run_dir_in_output_dir(run_dir: str):
    matches = list(Path(VLM_OUTPUT_DIR).rglob(run_dir))
    if len(matches) > 1:
        raise ValueError(f"Multiple matches found for run_dir '{run_dir}': {matches}")
    elif not matches:
        raise FileNotFoundError(f"No directory named '{run_dir}' found under {VLM_OUTPUT_DIR}")
    return matches[0]


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
