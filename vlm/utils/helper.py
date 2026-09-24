import os
import random
import string
from datetime import datetime
from typing import Any

import numpy as np
import torch


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


def add_datetime_tag(s: str, sep: str = "__") -> str:
    return datetime.now().strftime(f"{s}{sep}%Y-%m-%d_%H-%M-%S")


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
