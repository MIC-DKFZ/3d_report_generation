# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified to support VLM's 3D medical image conditioning and custom training workflow.

from typing import Optional

import torch
from transformers.generation import GenerationMixin


class VisionRopeGenerationMixin:
    """Generation position IDs for Qwen-VL models whose vision positions come from the adapter."""

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # Skip the upstream Qwen-VL override, which derives vision positions from the video grid.
        text_positions = GenerationMixin._prepare_position_ids_for_generation(
            self, inputs_tensor, model_kwargs
        )
        vision_positions, self.model.rope_deltas = self.model.get_rope_index(
            inputs_tensor,
            model_kwargs.get("vision_position_ids"),
            attention_mask=model_kwargs.get("attention_mask"),
        )
        # As upstream: [text; temporal; height; width]. Each decoding step continues every row
        # from its last position, which holds the maximum position after the vision blocks.
        return torch.cat([text_positions[None], vision_positions], dim=0)


def get_vision_rope_index(
    input_ids: Optional[torch.LongTensor],
    vision_position_ids: Optional[list[torch.Tensor]],
    attention_mask: Optional[torch.Tensor],
    video_token_id: int,
    vision_start_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the multimodal RoPE index (temporal, height, width) of each token.

    Adapted from the Qwen2.5-VL and Qwen3-VL 'get_rope_index': instead of deriving vision
    positions from the native video grid, each vision block uses the positions from
    'vision_position_ids' that its adapter produced. Text tokens continue after the largest
    preceding position.

    Returns the position IDs of shape (3, batch, sequence) and the per-sample offset between the
    largest position and the sequence length, which is used to continue positions while decoding.
    """
    mrope_position_deltas = []
    if input_ids is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        video_index = 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_videos = video_nums
            for _ in range(video_nums):
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                ed = ed_video

                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                current_vision_position_ids = vision_position_ids[video_index].to("cpu")
                llm_pos_ids_list.append(current_vision_position_ids + text_len + st_idx)
                st = ed + current_vision_position_ids.shape[1]

                video_index += 1
                remain_videos -= 1

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
