import glob
from typing import Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLTextModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    is_torchdynamo_compiling,
)

from configs.model.model_config import ModelConfig
from prep.evavit import get_config_from_plans
from registries.vision_encoder_registry import vision_encoder_registry
from utils import logger


class QwenVLModel3D(Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    config: Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config, model_cfg: dict):
        # we need to replace Qwen25VL init because it inits 'visual'
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.vision_encoder = None  # init in self.setup_vision_encoder()
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.rope_deltas = None     # cache rope_deltas here

        model_cfg = ModelConfig(**model_cfg)
        self.ve_cfg = model_cfg.vision_encoder

        # Initialize weights and apply final processing
        self.post_init()

        self.vision_adapter = model_cfg.adapter.to_adapter(vision_embed_dim=864, text_embed_dim=2048)
        # self.vision_adapter = VisionAdapterHeterogeneous(embedding_dim=864, embed_dim=1536, reduce_block=reduce_block)

    def setup_vision_encoder(self):
        # currently only working with MAE 
        # TODO implement other vision encoders
        vision_encoder_path = self.ve_cfg.vision_encoder_path
        plans_file = f"{vision_encoder_path}/plans.json"
        checkpoint_paths = glob.glob(f"{vision_encoder_path}/fold_all/checkpoint_*.pth")
        if len(checkpoint_paths) > 1:
            checkpoint_path = [c for c in checkpoint_paths if "final" in c][0]
            logger.info(f"Multiple VE checkpoints found, using final: {checkpoint_path}")
            if len(checkpoint_path) == 0:
                checkpoint_path = checkpoint_paths[0]
                logger.info(f"Multiple VE checkpoints found, using first: {checkpoint_path}")
        else:
            checkpoint_path = checkpoint_paths[0]

        config = get_config_from_plans(plans_file)
        vision_encoder_class = vision_encoder_registry[self.ve_cfg.vision_encoder_type]
        self.vision_encoder = vision_encoder_class(
            input_channels=1, output_channels=1, pretrained_weights_file=checkpoint_path, **config
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values_videos is not None:  # below two lines are updated by us
            ##########
            # Only these 3 lines (and bellow two lines) are different from the original Qwen2_5_VLModel
            #pixel_values_videos = torch.rand_like(pixel_values_videos) #TODO RANDOM IMAGE HERE!!!
            #pixel_values_videos = pixel_values_videos.to(self.vision_adapter.adapter[0].weight.dtype)
            if hasattr(self, "vision_encoder") and self.vision_encoder is not None:
                video_embeds = self.vision_encoder(pixel_values_videos.unsqueeze(1))
                #video_embeds = rearrange(video_embeds, 'b c h w d -> b c (h w d )')
                video_embeds = self.vision_adapter(video_embeds, video_grid_thw)
            else:
                video_embeds = self.vision_adapter(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.view(-1, video_embeds.shape[-1])  # Flatten to (total_features, embed_dim)
            ##########
            if input_ids is None:
                video_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                video_mask = video_mask.all(-1)
            else:
                video_mask = input_ids == self.config.video_token_id

            n_video_tokens = (video_mask).sum()
            n_video_features = video_embeds.shape[0]
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            #TODO this fails for images with different sizes when batch size > 1!
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)


        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                #########
                # This loop is also added by us
                for video in video_grid_thw:
                    # We multiply dimension 1 (height) and dimension 2 (width) by 2 in order to nullify the division
                    # by the spatial_merge_size=2 in the get_rope_index() function below
                    video[1] = video[1]*2
                    video[2] = video[2]*2
                #########
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                #########
                #TODO this also added and not in original Qwen. Without it I get crash in val. why do we need this?
                position_ids = position_ids.clone()
                #########
                position_ids += delta.to(position_ids.device)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()
    


class QwenVLForConditionalGeneration3D(Qwen2_5_VLForConditionalGeneration, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual|vision_adapter|vision_encoder))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, model_cfg: dict):
        super().__init__(config)
        self.model = QwenVLModel3D(config, model_cfg=model_cfg)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepared with rope_deltas
        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            if cache_position[0] == 0 or self.model.rope_deltas is None:
                 #########
                # wieder wie oben... just this part changed
                video_grid_thw_for_rope = video_grid_thw.clone() if video_grid_thw is not None else None
                for video in video_grid_thw_for_rope:
                    # Since we use the time dimension as spatial dimension we have to apply the reduce factor already here since it is not done in the rope function below
                    # For reduceblocks different to 2,2,2 we have to also adjus dim 1 and 2
                    video[0] = torch.floor_divide(video[0], torch.tensor(self.reduce_block[0], device=video.device, dtype=video.dtype))
                    video[1] = torch.floor_divide(video[1], torch.tensor(self.reduce_block[1] // 2, device=video.device, dtype=video.dtype))
                    video[2] = torch.floor_divide(video[2], torch.tensor(self.reduce_block[2] // 2, device=video.device, dtype=video.dtype))
                #########
                vision_positions, rope_deltas = self.model.get_rope_index(
                    model_inputs.get("input_ids", None),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw_for_rope,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            elif "position_ids" in model_inputs:
                position_ids = model_inputs["position_ids"][None, ...]
                delta = self.model.rope_deltas
                delta = delta.repeat_interleave(position_ids.shape[1] // delta.shape[0], dim=0)
                vision_positions = position_ids + delta.expand_as(position_ids)
                vision_positions = vision_positions.expand(3, vision_positions.shape[1], -1)

            # Concatenate "text + vision" positions into [4, bs, seq-len]
            if "position_ids" not in model_inputs:
                text_positions = torch.arange(input_ids, device=input_ids.device)[None, None, :]
            else:
                text_positions = model_inputs["position_ids"][None, ...]
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

