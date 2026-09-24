from __future__ import annotations

import random

import numpy as np


def should_apply(p_apply: float) -> bool:
    return random.random() < p_apply


def validate_probability(name: str, value: float):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def validate_embedding(embedding) -> np.ndarray:
    if isinstance(embedding, np.ndarray):
        return embedding
    raise TypeError(f"Unsupported embedding type: {type(embedding).__name__}")


def clone_embedding(embedding) -> np.ndarray:
    return validate_embedding(embedding).copy()
