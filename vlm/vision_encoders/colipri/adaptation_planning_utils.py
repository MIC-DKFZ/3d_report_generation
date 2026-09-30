from functools import lru_cache

import torch


@lru_cache(maxsize=4)
def _load_ckpt(ckpt_path: str) -> dict:
    """
    Wrapper to cache the loading of checkpoint files, to minimize I/O requests.

    Args:
        ckpt_path (str): The path to the checkpoint file we want to load

    Returns:
        dict: The loaded checkpoint dictionary
    """
    return torch.load(ckpt_path, map_location="cpu")


def mb_load_ckpt(ckpt_path: str, ckpt: dict | None = None, network_key: str | None = "network_weights") -> dict:
    """
    Loads a checkpoint from the given path or dictionary, if the ckpt is not already given

    Args:
        ckpt_path (str): Path to the checkpoint file.
        ckpt (dict, optional): Already loaded checkpoint dictionary. Defaults to None.

    Returns:
        dict: Loaded checkpoint dictionary.
    """
    if ckpt is None:
        ckpt = _load_ckpt(ckpt_path)
    if network_key is None:
        return ckpt
    return ckpt[network_key]
