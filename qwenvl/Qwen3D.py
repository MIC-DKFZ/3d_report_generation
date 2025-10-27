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
    Qwen2_5_VLTextModel, KwargsForCausalLM,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
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
            logger.info_main(f"Multiple VE checkpoints found, using final: {checkpoint_path}")
            if len(checkpoint_path) == 0:
                checkpoint_path = checkpoint_paths[0]
                logger.info_main(f"Multiple VE checkpoints found, using first: {checkpoint_path}")
        else:
            checkpoint_path = checkpoint_paths[0]

        config = get_config_from_plans(plans_file)
        vision_encoder_class = vision_encoder_registry[self.ve_cfg.type]
        self.vision_encoder = vision_encoder_class(
            input_channels=1, output_channels=1, pretrained_weights_file=checkpoint_path, **config
        )

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        From 'Qwen2_5_VLModel.get_rope_index' adjusted to our usage.

        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        # spatial_merge_size = self.config.vision_config.spatial_merge_size
        # image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        # if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None): #
        if input_ids is not None and video_grid_thw is not None:
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
            image_index, video_index = 0, 0 #
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                # image_nums, video_nums = 0, 0 #
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                # image_nums = (vision_tokens == image_token_id).sum() #
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                # remain_images, remain_videos = image_nums, video_nums #
                # for _ in range(image_nums + video_nums): #
                remain_videos = video_nums
                for _ in range(video_nums):
                    # if image_token_id in input_tokens and remain_images > 0: #
                    #     ed_image = input_tokens.index(image_token_id, st) #
                    # else:
                    #     ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    # if ed_image < ed_video: #
                    #     t, h, w = (
                    #         image_grid_thw[image_index][0],
                    #         image_grid_thw[image_index][1],
                    #         image_grid_thw[image_index][2],
                    #     )
                    #     second_per_grid_t = 0
                    #     image_index += 1
                    #     # remain_images -= 1
                    #     ed = ed_image
                    #
                    # else:
                    # t, h, w = (
                    #     video_grid_thw[video_index][0],
                    #     video_grid_thw[video_index][1],
                    #     video_grid_thw[video_index][2],
                    # )
                    llm_grid_t, llm_grid_h, llm_grid_w = video_grid_thw[video_index]
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                    # llm_grid_t, llm_grid_h, llm_grid_w = (
                    #     t.item(),
                    #     h.item() // spatial_merge_size,
                    #     w.item() // spatial_merge_size,
                    # )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    ## normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
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
        **kwargs: Unpack[KwargsForCausalLM],
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

            if pixel_values_videos is not None:
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
                n_video_tokens = (input_ids == self.config.video_token_id).sum()
                n_video_features = video_embeds.shape[0]
                if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

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
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
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
