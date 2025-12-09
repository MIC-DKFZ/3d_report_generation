from abc import ABC

import torch

from vlm.adapters.base import AdapterBase
from vlm.utils.vision_input_type import VisionInputType


class QueryAdapter(AdapterBase, ABC):
    def __init__(self, num_query_tokens: int, **kwargs):
        super().__init__(**kwargs)
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

