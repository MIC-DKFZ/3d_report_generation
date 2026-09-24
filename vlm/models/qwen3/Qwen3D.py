from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model

from vlm.configs.model.model_config import ModelConfig


class Qwen3TextVLModel3D(nn.Module):
    def __init__(
        self,
        config,
        model_cfg: dict,
        vision_embed_dim: int,
        vision_embed_size: tuple[int, int, int],
        language_model: Qwen3Model | None = None,
    ):
        super().__init__()
        self.config = config
        self.language_model = language_model or Qwen3Model(config)
        self.vision_adapter = None
        self.vision_encoder = None

        self.model_cfg = ModelConfig(**model_cfg)
        self.vision_embed_dim = vision_embed_dim
        self.vision_embed_size = vision_embed_size
        self.text_embed_dim: int = config.hidden_size

    def setup_vision_adapter(self):
        self.vision_adapter = self.model_cfg.build_adapter(
            vision_embed_dim=self.vision_embed_dim,
            vision_embed_size=self.vision_embed_size,
            text_embed_dim=self.text_embed_dim,
        )

    def setup_vision_encoder(self):
        self.vision_encoder = self.model_cfg.build_vision_encoder()

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
            raise ValueError("Qwen3 text VLM config is missing 'video_token_id'.")

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
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
                raise ValueError("input_ids are required when inserting Qwen3 vision embeddings")

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
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )


class Qwen3TextVLForConditionalGeneration(Qwen3ForCausalLM, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        model_cfg: dict,
        vision_embed_dim: int,
        vision_embed_size: tuple[int, int, int],
        **kwargs,
    ):
        pretrained_qwen = Qwen3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        model = cls(
            pretrained_qwen.config,
            model_cfg=model_cfg,
            vision_embed_dim=vision_embed_dim,
            vision_embed_size=vision_embed_size,
        )
        model.model.language_model = pretrained_qwen.model
        model.lm_head = pretrained_qwen.lm_head
        model.generation_config = pretrained_qwen.generation_config
        return model

    def __init__(
        self,
        config,
        model_cfg: dict,
        vision_embed_dim: int,
        vision_embed_size: tuple[int, int, int],
    ):
        super().__init__(config)
        language_model = self.model
        self.model = Qwen3TextVLModel3D(
            config,
            model_cfg=model_cfg,
            vision_embed_dim=vision_embed_dim,
            vision_embed_size=vision_embed_size,
            language_model=language_model,
        )

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
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: Optional[bool] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        vision_position_ids: Optional[list[torch.Tensor]] = None,
        adapter_args: Optional[dict] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            return_dict=True if return_dict is None else return_dict,
            pixel_values_videos=pixel_values_videos,
            vision_position_ids=vision_position_ids,
            adapter_args=adapter_args,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
