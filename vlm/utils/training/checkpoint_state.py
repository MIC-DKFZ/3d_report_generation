"""Partial weight checkpoints. The caller constructs the base model before loading."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from safetensors.torch import load_file, save_file

from vlm.utils import logger

MANIFEST_NAME = "checkpoint_manifest.json"
WEIGHTS_NAME = "model.safetensors"


@dataclass
class CheckpointState:
    required_keys: set[str] = field(default_factory=set)
    previous_run_dirs: list[str] = field(default_factory=list)


def ensure_checkpoint_state(model) -> CheckpointState:
    state = getattr(model, "_vlm_checkpoint_state", None)
    if state is None:
        state = CheckpointState()
        model._vlm_checkpoint_state = state
    return state


def record_trainable_checkpoint_keys(model) -> CheckpointState:
    state = ensure_checkpoint_state(model)
    state.required_keys.update(
        name for name, param in model.named_parameters(remove_duplicate=False) if param.requires_grad
    )
    return state


def initialize_checkpoint_state(model):
    """Record currently trainable parameters and persistent buffers."""
    tracking = record_trainable_checkpoint_keys(model)
    parameters = dict(model.named_parameters(remove_duplicate=False))

    # Include persistent buffers, which appear in state_dict but are not parameters.
    tracking.required_keys.update(set(model.state_dict()) - set(parameters))

    return tracking


def read_manifest(directory):
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def checkpoint_state_dict(model, state_dict=None):
    required = record_trainable_checkpoint_keys(model).required_keys
    state = model.state_dict() if state_dict is None else state_dict

    missing = required - state.keys()
    if missing:
        raise ValueError(f"Required checkpoint keys missing from model state: {sorted(missing)}")

    return {name: state[name] for name in sorted(required)}


def save_model_checkpoint(model, directory, *, state_dict=None):
    selected = checkpoint_state_dict(model, state_dict)
    tracking = ensure_checkpoint_state(model)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Independent copies let safetensors store shared parameters under each required name.
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in selected.items()}
    save_file(tensors, directory / WEIGHTS_NAME)
    with (directory / MANIFEST_NAME).open("w", encoding="utf-8") as stream:
        json.dump({"previous_run_dirs": tracking.previous_run_dirs}, stream, indent=2)
        stream.write("\n")


def load_model_checkpoint(model, directory):
    directory = Path(directory)
    weights = load_file(directory / WEIGHTS_NAME)
    manifest = read_manifest(directory)
    tracking = ensure_checkpoint_state(model)
    if manifest is None:
        logger.warning_main(
            "Loading a legacy checkpoint without a manifest; completeness cannot be guaranteed."
        )
        previous_run_dirs = tracking.previous_run_dirs
    else:
        previous_run_dirs = manifest["previous_run_dirs"]

    state = model.state_dict()
    for name, value in weights.items():
        if name not in state:
            raise ValueError(f"Loaded checkpoint key does not exist in model: {name}")
        if value.shape != state[name].shape or value.dtype != state[name].dtype:
            raise ValueError(f"Checkpoint tensor shape/dtype mismatch for {name}")

    result = model.load_state_dict(weights, strict=False)
    previous_run_dirs = list(previous_run_dirs)
    source_run_dir = str(directory.parent)
    if not previous_run_dirs or previous_run_dirs[-1] != source_run_dir:
        previous_run_dirs.append(source_run_dir)
    tracking.previous_run_dirs = previous_run_dirs
    tracking.required_keys.update(weights)
    return result.missing_keys, result.unexpected_keys
