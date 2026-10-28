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

"""Shared integration of adapter vision embeddings into the Qwen-VL model families."""

from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple


class AdapterVisionModelMixin:
    """
    Inner Qwen-VL model whose vision embeddings come from the adapter.

    Concrete classes put this mixin before the upstream model class, skip the native vision tower
    in '__init__', and set 'output_class' to the upstream model output class.
    """

    output_class: type

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        vision_position_ids: Optional[list[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the RoPE index from the adapter's vision positions instead of the video grid."""
        return get_vision_rope_index(
            input_ids,
            vision_position_ids,
            attention_mask,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        vision_position_ids: Optional[list[torch.Tensor]] = None,
        adapter_args: Optional[dict] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values_videos is not None:
            video_embeds = pixel_values_videos
            if self.vision_encoder is not None:
                video_embeds = self.vision_encoder(video_embeds)
            if self.vision_adapter is not None:
                video_embeds = self.vision_adapter(video_embeds, adapter_args)
            video_embeds = video_embeds.view(-1, video_embeds.shape[-1])
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    vision_position_ids,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (past_key_values_length + self.rope_deltas).to(inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return self.output_class(**outputs, rope_deltas=self.rope_deltas)


class AdapterVisionForConditionalGenerationMixin:
    """
    Qwen-VL generation model whose vision embeddings come from the adapter.

    Concrete classes put this mixin before the upstream model class and set 'output_class' to the
    upstream causal LM output class.
    """

    output_class: type
    # Report the loss over the summed tokens of the accumulation window ('num_items_in_batch').
    accepts_loss_kwargs = True
    # The adapter replaces the native vision tower stored in the hub checkpoints.
    _keys_to_ignore_on_load_unexpected = [r"^model\.visual\."]

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

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        vision_position_ids: Optional[list[torch.Tensor]] = None,
        adapter_args: Optional[dict] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """
        Our additional forward kwargs 'vision_position_ids' and 'adapter_args' are passed on via the kwargs argument,
        so we wouldn't have to override this function. However, the GenerationMixin._validate_model_kwargs() function
        looks for unused forward arguments by inspecting the signature of this forward function, hence we add
        'vision_position_ids' and 'adapter_args' to the function signature.
        """

        outputs = self.model(
            input_ids=input_ids,
            pixel_values_videos=pixel_values_videos,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_position_ids=vision_position_ids,
            adapter_args=adapter_args,
            **kwargs,
        )

        hidden_states = outputs[0]

        loss = None
        if labels is not None:
            # Apply the LM head only where a target exists (the response), skipping the vision and prompt
            # positions. Logits are then (num_targets, vocab) instead of (batch, sequence, vocab).
            shift_labels = labels[:, 1:]
            has_target = shift_labels != -100
            logits = self.lm_head(hidden_states[:, :-1][has_target])
            loss = self.loss_function(
                logits=logits,
                labels=None,
                shift_labels=shift_labels[has_target],
                vocab_size=self.config.text_config.vocab_size,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        return self.output_class(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


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
