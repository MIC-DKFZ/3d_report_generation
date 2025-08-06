import os
import random
from argparse import ArgumentParser

import numpy as np
import torch

from utils import logger


def set_seed(seed):
    seed = int(seed)
    logger.info(f"SETTING GLOBAL SEED TO {seed}")
    # pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
