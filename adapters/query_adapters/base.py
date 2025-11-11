from abc import ABC

import torch

from adapters.base import AdapterBase
from utils.vision_input_type import VisionInputType


class QueryAdapter(AdapterBase, ABC):
    def __init__(
        self,
        vision_embed_dim: int,
        text_embed_dim: int,
        num_query_tokens: int
    ):
        super().__init__(vision_embed_dim=vision_embed_dim, text_embed_dim=text_embed_dim)
        self.num_query_tokens = num_query_tokens

    def get_adapter_args(
            self,
            vision_inputs: list[torch.Tensor],
            vision_input_type: VisionInputType,
            **kwargs
    ) -> tuple[list[int], list[torch.Tensor], dict]:
        num_vision_tokens = ...
        vision_position_ids = ...

        return num_vision_tokens, vision_position_ids, {}

