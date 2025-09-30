import gc
import json
import os
import re
from typing import Any, Iterable

import torch
from functools import partial
from safetensors.torch import load_file as safe_load_file
from utils import logger
from transformers import is_safetensors_available
from transformers.utils import WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME, check_torch_load_is_safe


def apply_regex_subs(
        keys_or_dict: list[str] | dict[str, Any], key_mapping: dict[str, str]
) -> list[str] | dict[str, Any]:
    """
    Apply regex substitutions to keys in a list or dict.

    Args:
        keys_or_dict: Either a list of keys or a dict with string keys.
        key_mapping: A dict of regex pattern -> replacement string. Patterns are applied in insertion order.

    Returns:
        A new list of keys or dict with updated keys.
    """
    is_dict = isinstance(keys_or_dict, dict)
    new_keys_or_dict = {} if is_dict else []

    for key in keys_or_dict:
        new_key = key
        for pattern, replacement in key_mapping.items():
            new_key = re.sub(pattern, replacement, new_key)
        if is_dict:
            new_keys_or_dict[new_key] = keys_or_dict[key]
        else:
            new_keys_or_dict.append(new_key)

    return new_keys_or_dict


def assert_keys_equality(loaded_keys: Iterable[str], model_keys: Iterable[str], keys_to_ignore: Iterable[str]) -> None:
    missing_keys = [key for key in model_keys if key not in loaded_keys
                    if key not in keys_to_ignore]   # added second if condition
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        error_message = f"Error(s) in loading state_dict"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s) in loaded dict: {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nUnexpected key(s) from loaded dict: {str_unexpected_keys}."
        raise RuntimeError(error_message)


def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True,
                            key_mapping: dict | None = None,
                            keys_to_ignore: list | None = None) -> None:
    """
    Adapted from 'transformers.modeling_utils.load_sharded_checkpoint()',
    with optional key remapping (e.g. from '_checkpoint_conversion_mapping') and keys to ignore.
    """
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # load safe due to preference
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True  # load safe since we have no other choice

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = index["weight_map"].keys()

    loaded_keys = apply_regex_subs(list(loaded_keys), key_mapping)  # added line
    model_keys = model.state_dict().keys()
    if strict:
        assert_keys_equality(loaded_keys, model_keys, keys_to_ignore)   # outsourced to assert_keys_equality()

    if load_safe:
        loader = safe_load_file
    else:
        check_torch_load_is_safe()
        loader = partial(torch.load, map_location="cpu", weights_only=True)

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        state_dict = apply_regex_subs(state_dict, key_mapping)  # added line
        model.load_state_dict(state_dict, strict=False)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()
