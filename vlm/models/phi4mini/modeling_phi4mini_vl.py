# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.phi3.modeling_phi3 import (
    Phi3ForCausalLM,
    Phi3Model,
    Phi3PreTrainedModel,
)


class Phi4MiniVLModel(nn.Module):
    def __init__(
        self,
        config,
        language_model: Phi3Model | None = None,
    ):
        super().__init__()
        self.config = config
        self.language_model = language_model if language_model is not None else Phi3Model(config)
        self.vision_adapter = None
        self.vision_encoder = None

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        video_features: torch.FloatTensor,
    ) -> torch.BoolTensor:
        video_token_id = getattr(self.config, "video_token_id", None)
        if video_token_id is None:
            raise ValueError("Phi VLM config is missing 'video_token_id'.")

        n_video_tokens = (input_ids == video_token_id).sum()
        n_video_features = video_features.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                "Video features and video tokens do not match: "
                f"tokens: {n_video_tokens}, features {n_video_features}"
            )

        return (input_ids == video_token_id).unsqueeze(-1).expand_as(inputs_embeds)

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
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify either input_ids or inputs_embeds")
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values_videos is not None:
            if input_ids is None:
                raise ValueError("input_ids are required when inserting Phi vision embeddings")

            video_embeds = pixel_values_videos
            if self.vision_encoder is not None:
                video_embeds = self.vision_encoder(video_embeds)
            if self.vision_adapter is not None:
                video_embeds = self.vision_adapter(video_embeds, adapter_args or {})

            video_embeds = video_embeds.view(-1, video_embeds.shape[-1])
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.to(inputs_embeds.device),
                video_embeds,
            )

        return self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


class Phi4MiniVLForConditionalGeneration(Phi3ForCausalLM, GenerationMixin):
    accepts_loss_kwargs = True
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        **kwargs,
    ):
        pretrained_phi = Phi3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        model = cls(
            pretrained_phi.config,
            language_model=pretrained_phi.model,
            lm_head=pretrained_phi.lm_head,
        )
        model.generation_config = pretrained_phi.generation_config
        return model

    def __init__(
        self,
        config,
        language_model: Phi3Model | None = None,
        lm_head: nn.Linear | None = None,
    ):
        # Initialize HF metadata without constructing a second decoder.
        Phi3PreTrainedModel.__init__(self, config)
        self.model = Phi4MiniVLModel(config, language_model=language_model)
        self.vocab_size = config.vocab_size
        self.lm_head = (
            lm_head
            if lm_head is not None
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model.language_model

    def set_decoder(self, decoder):
        self.model.language_model = decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        vision_position_ids: Optional[list[torch.Tensor]] = None,
        adapter_args: Optional[dict] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Run the upstream forward; the vision kwargs are consumed by the inner model. They are listed
        here because GenerationMixin._validate_model_kwargs() inspects this signature.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            logits_to_keep=logits_to_keep,
            pixel_values_videos=pixel_values_videos,
            vision_position_ids=vision_position_ids,
            adapter_args=adapter_args,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        pixel_values_videos=None,
        vision_position_ids=None,
        adapter_args=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        is_prefill = past_key_values is None
        if cache_position is not None:
            is_prefill = bool(cache_position[0].item() == 0)

        if is_prefill:
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["vision_position_ids"] = vision_position_ids
            model_inputs["adapter_args"] = adapter_args

        return model_inputs
