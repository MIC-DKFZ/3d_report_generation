import torch.nn as nn
import torch
import numpy as np
from abc import ABC, abstractmethod

from utils.vision_input_type import VisionInputType


class AdapterBase(nn.Module, ABC):
    def __init__(
        self,
        vision_embed_dim: int,
        text_embed_dim: int
    ):
        super().__init__()
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim

    @abstractmethod
    def get_adapter_args(
            self,
            vision_inputs: list[np.ndarray],
            vision_input_type: VisionInputType,
            **kwargs
    ) -> tuple[list[int], list[torch.Tensor], dict]:
        """
        Returns the number of vision tokens, the vision_position_ids and
        optional args for the adapter forward pass for a batch of samples.
        """
        ...

