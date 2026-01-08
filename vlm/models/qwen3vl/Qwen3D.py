import glob
from typing import Optional, Union

import torch
from torch import nn
from transformers.generation import GenerationMixin
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel, Qwen3VLCausalLMOutputWithPast,
)
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils.generic import check_model_inputs
from transformers.utils import (
    is_torchdynamo_compiling, can_return_tuple,
    TransformersKwargs, auto_docstring
)

from vlm.configs.model.model_config import ModelConfig
from vlm.utils.helper import get_config_from_plans
from vlm.registries.vision_encoder_registry import vision_encoder_registry
from vlm.utils import logger


class QwenVLModel3D(Qwen3VLModel, Qwen3VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config, model_cfg: dict, vision_embed_dim: int, vision_embed_size: tuple[int, int, int]):
        # we need to replace Qwen25VL init because it inits 'visual'
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.vision_adapter = None  # init in self.setup_vision_adapter()
        self.vision_encoder = None  # init in self.setup_vision_encoder()
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None     # cache rope_deltas here

        self.model_cfg = ModelConfig(**model_cfg)

        self.vision_embed_dim = vision_embed_dim
        self.vision_embed_size = vision_embed_size
        self.text_embed_dim: int = config.text_config.hidden_size

        # Initialize weights and apply final processing
        # self.post_init()

    def setup_vision_adapter(self):
        self.vision_adapter = self.model_cfg.build_adapter(
            vision_embed_dim=self.vision_embed_dim,
            vision_embed_size=self.vision_embed_size,
            text_embed_dim=self.text_embed_dim
        )

    def setup_vision_encoder(self):
        vision_encoder_path = self.model_cfg.vision_encoder.vision_encoder_path
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
        vision_encoder_class = vision_encoder_registry[self.model_cfg.vision_encoder.type]
        self.vision_encoder = vision_encoder_class(
            input_channels=1, output_channels=1, pretrained_weights_file=checkpoint_path, **config
        )

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        vision_position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        # if video_grid_thw is not None:
        #     video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        #     video_grid_thw[:, 0] = 1

        # spatial_merge_size = self.config.vision_config.spatial_merge_size
        # image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        # if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
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
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                # image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                # image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                # remain_images, remain_videos = image_nums, video_nums
                remain_videos = video_nums
                for _ in range(video_nums):
                    # if image_token_id in input_tokens and remain_images > 0:
                    #     ed_image = input_tokens.index(image_token_id, st)
                    # else:
                    #     ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    # if ed_image < ed_video:
                    #     t, h, w = (
                    #         image_grid_thw[image_index][0],
                    #         image_grid_thw[image_index][1],
                    #         image_grid_thw[image_index][2],
                    #     )
                    #     image_index += 1
                    #     remain_images -= 1
                    #     ed = ed_image

                    # else:
                    #     t, h, w = (
                    #         video_grid_thw[video_index][0],
                    #         video_grid_thw[video_index][1],
                    #         video_grid_thw[video_index][2],
                    #     )
                    #     video_index += 1
                    #     remain_videos -= 1
                    #     ed = ed_video
                    # llm_grid_t, llm_grid_h, llm_grid_w = (
                    #     t.item(),
                    #     h.item() // spatial_merge_size,
                    #     w.item() // spatial_merge_size,
                    # )

                    ed = ed_video
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    # t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    # h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    # w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    # llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    current_vision_position_ids = vision_position_ids[video_index].to("cpu")
                    llm_pos_ids_list.append(current_vision_position_ids + text_len + st_idx)
                    # st = ed + llm_grid_t * llm_grid_h * llm_grid_w
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

    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,

        vision_position_ids: Optional[list[torch.Tensor]] = None,
        adapter_args: Optional[dict] = None,

        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            ######
            # Changing upstream implementation here
            video_embeds = pixel_values_videos
            if self.vision_encoder is not None:
                video_embeds = self.vision_encoder(video_embeds)
            if self.vision_adapter is not None:
                video_embeds = self.vision_adapter(video_embeds, adapter_args)
            video_embeds = video_embeds.view(-1, video_embeds.shape[-1])  # Flatten to (total_features, embed_dim)
            ######

            # video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            # video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            # deepstack_visual_embeds = deepstack_video_embeds

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
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )



class QwenVLForConditionalGeneration3D(Qwen3VLForConditionalGeneration, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    config: Qwen3VLConfig
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config, model_cfg: dict, vision_embed_dim: int, vision_embed_size: tuple[int, int, int]):
        super().__init__(config)
        self.model = QwenVLModel3D(
            config,
            model_cfg=model_cfg,
            vision_embed_dim=vision_embed_dim,
            vision_embed_size=vision_embed_size
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,

        vision_position_ids: Optional[list[torch.Tensor]] = None,
        adapter_args: Optional[dict] = None,

        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        """
        Our additional forward kwargs 'vision_position_ids' and 'adapter_kwargs' are passed on via the kwargs argument,
        so we wouldn't have to override this function. However, the GenerationMixin._validate_model_kwargs() function
        looks for unused forward arguments by inspecting the signature of this forward function, hence we add
        'vision_position_ids' and 'adapter_kwargs' to the function signature.
        """

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
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
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )