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

from typing import Optional, Union

import torch
from torch import nn
from transformers.generation import GenerationMixin
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration,
    Qwen3VLModel as HFQwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel, Qwen3VLCausalLMOutputWithPast,
)
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils.generic import check_model_inputs
from transformers.utils import can_return_tuple, TransformersKwargs

from vlm.models.rope import get_vision_rope_index


class Qwen3VLModel(HFQwen3VLModel, Qwen3VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        # The HF model constructor also creates the unused native vision tower.
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.vision_adapter = None
        self.vision_encoder = None
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None     # cache rope_deltas here
        self.post_init()

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

    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_position_ids: Optional[list[torch.Tensor]] = None,
        adapter_args: Optional[dict] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        visual_pos_masks = None
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
            visual_pos_masks = video_mask[..., 0]

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
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=None,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )



class Qwen3VLForConditionalGeneration(HFQwen3VLForConditionalGeneration, GenerationMixin):
    accepts_loss_kwargs = True
    _checkpoint_conversion_mapping = {}
    config: Qwen3VLConfig

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        vision_position_ids: Optional[list[torch.Tensor]] = None,
        adapter_args: Optional[dict] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
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
            cache_position=cache_position,
            vision_position_ids=vision_position_ids,
            adapter_args=adapter_args,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                num_items_in_batch=num_items_in_batch,
            )

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
