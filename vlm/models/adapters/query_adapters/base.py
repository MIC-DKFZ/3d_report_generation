from abc import ABC

import torch

from vlm.models.adapters.base import AdapterBase
from vlm.utils.data.vision_input_type import VisionInputType


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

        batch_size = len(vision_inputs)
        num_vision_tokens = [self.num_query_tokens] * batch_size

        template = torch.arange(self.num_query_tokens).unsqueeze(0).repeat(3, 1)
        vision_position_ids = [template.clone() for _ in range(batch_size)]

        return num_vision_tokens, vision_position_ids, {}
