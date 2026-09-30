import torch.nn as nn
import torch
import numpy as np
from abc import ABC, abstractmethod

from vlm.registries.norm_layer_registry import NORM_LAYER_REGISTRY
from vlm.utils.class_utils import get_class_from_registry
from vlm.utils.data.vision_input_type import VisionInputType


class AdapterBase(nn.Module, ABC):
    def __init__(
        self,
        vision_embed_dim: int,
        text_embed_dim: int,
        num_vision_tokens: int,
        bias: bool = True,
        norm_layer_class_name: str = "RMSNorm"
    ):
        super().__init__()
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        self.num_vision_tokens = num_vision_tokens
        self.bias = bias
        self.norm_layer_class = get_class_from_registry(norm_layer_class_name, NORM_LAYER_REGISTRY)

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
