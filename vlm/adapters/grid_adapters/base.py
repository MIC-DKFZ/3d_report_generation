from abc import ABC
import torch
from vlm.adapters.base import AdapterBase
from vlm.utils.vision_input_type import VisionInputType
import numpy as np


class GridAdapter(AdapterBase, ABC):
    """
    Adapter for processing homogeneously-sized image batches.
    This class retains the spatial grid structure of the vision embeddings and implements M-RoPE.
    """
    def __init__(
            self,
            vision_embed_dim: int,
            text_embed_dim: int,
            reduce_block: tuple[int, int, int]
    ):
        super().__init__(vision_embed_dim, text_embed_dim)
        self.reduce_block = np.array(reduce_block)

    def get_adapter_args(
            self,
            vision_inputs: list[np.ndarray],
            vision_input_type: VisionInputType,
            **kwargs
    ) -> tuple[list[int], list[torch.Tensor], dict]:

        batch_size = len(vision_inputs)

        grid = np.array(vision_inputs[0].shape[-3:])  # take first one since all vision inputs are of same shape
        if vision_input_type == VisionInputType.IMAGE:
            downsample_factor = np.array(kwargs["downsample_factor"])
            grid = grid // downsample_factor
        grid = grid // self.reduce_block
        num_vision_tokens = [int(grid.prod().item())] * batch_size
        video_grid_thw = grid[None, ...].repeat(batch_size, axis=0)

        video_grid_thw = torch.from_numpy(video_grid_thw).int()

        # Calc vision_position_ids
        llm_grid_t, llm_grid_h, llm_grid_w = grid

        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

        # time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
        time_tensor = expanded_range

        time_tensor_long = time_tensor.long()
        t_index = time_tensor_long.flatten()

        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

        template = torch.stack([t_index, h_index, w_index])
        vision_position_ids = [template.clone() for _ in range(batch_size)]

        return num_vision_tokens, vision_position_ids, {"video_grid_thw": video_grid_thw}


class GridAdapterHeterogenous(AdapterBase, ABC):
    """
    Adapter for processing heterogeneously-sized image batches.
    This class retains the spatial grid structure of the vision embeddings and implements M-RoPE.
    """
    def __init__(
            self,
            vision_embed_dim: int,
            text_embed_dim: int,
            reduce_block: tuple[int, int, int]
    ):
        super().__init__(vision_embed_dim, text_embed_dim)
        self.reduce_block = np.array(reduce_block)

    def get_adapter_args(
            self,
            vision_inputs: list[torch.Tensor],
            vision_input_type: VisionInputType,
            **kwargs
    ) -> tuple[list[int], list[torch.Tensor], dict]:
        if vision_input_type == VisionInputType.EMBEDDING:
            video_grid_thw = np.stack(
                [
                    np.array(vision_input.shape[-3:]) // np.array(self.reduce_block) for vision_input in
                    vision_inputs
                ]
            )

        elif vision_input_type == VisionInputType.IMAGE:
            downsample_factor: tuple[int, int, int] = kwargs["downsample_factor"]
            ds = np.array(downsample_factor)
            video_grid_thw = np.stack(
                [
                    np.array(vision_input.shape[-3:]) // ds // self.reduce_block for vision_input in vision_inputs
                ]
            )
        else:
            raise ValueError(f"Unknown vision input type: {vision_input_type}")
        video_grid_thw = torch.from_numpy(video_grid_thw).int()

        num_vision_tokens = []
        for grid in video_grid_thw:
            num_vision_tokens.append(grid.prod().item())

        position_ids = ...

        return num_vision_tokens, position_ids, {"video_grid_thw": video_grid_thw}

