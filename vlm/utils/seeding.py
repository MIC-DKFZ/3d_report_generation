"""Random seeding shared by training and inference."""

import random

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
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # The two lines below might be needed for full reproducibility with CUDA,
        # but can impact performance. Use them if you absolutely need deterministic behavior.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
