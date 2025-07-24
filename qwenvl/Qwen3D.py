from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from transformers.activations import QuickGELUActivation

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VLModelOutputWithPast, Qwen2RMSNorm
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from  transformers.utils import auto_docstring, can_return_tuple, TransformersKwargs, is_torchdynamo_compiling, logging

#from transformers.models.qwen2_vl.modeling_qwen2_vl

class VisionAdapter(nn.Module):
    def __init__(self, embedding_dim: int, embed_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, embed_dim),  # 864 -> 1536
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),  # 1536 -> 1536
        )

    def forward(self, x):
        # x shape: (batch_size, 864, sequence_length)
        # Need to output: (sequence_length, embed_dim)

        # Transpose to get sequence_length as first dimension
        x = x.transpose(1, 2)  # -> (batch_size, sequence_length, 864)

        # Reshape to process all sequences at once
        batch_size, seq_len, emb_dim = x.shape
        x = x.reshape(-1, emb_dim)  # -> (batch_size * sequence_length, 864)

        # Apply adapter
        x = self.adapter(x)  # -> (batch_size * sequence_length, 1536)

        # Reshape back
        x = x.reshape(
            batch_size * seq_len, -1
        )  # -> (batch_size * sequence_length, 1536)

        return x


class VisionAdapterReduceBlocks(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        embed_dim: int,
        reduce_block: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        self.reduce_block = reduce_block

        hidden_size = embedding_dim * reduce_block[0] * reduce_block[1] * reduce_block[2]
        self.ln_q = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.adapter = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(hidden_size, embed_dim)
        )

    def forward(self, x, video_grid_thw):
        # Convert input to float first
        x = x.to(torch.bfloat16)

        # Input shape is (batch_size, embedding_dim, sequence_length)
        batch_size, embedding_dim, _ = x.shape

        # TODO how to handle cases where each image has different size i.e. video_grid_thw has size (B, 3) and this only works for B==1?
        x = x.reshape(batch_size, embedding_dim, *video_grid_thw[0])

        # Use unfold to create reduce_block
        x = x.unfold(2, self.reduce_block[0], self.reduce_block[0])
        x = x.unfold(3, self.reduce_block[1], self.reduce_block[1])
        x = x.unfold(4, self.reduce_block[2], self.reduce_block[2])

        # Reshape to combine all reduce_block
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7)  # Move embedding_dim after spatial dims
        # (B, T', H', W', C, rT, rH, rW) -> (B, T'*H'*W', C*rT*rH*rW)
        x = x.reshape(
            batch_size,
            -1,
            embedding_dim
            * self.reduce_block[0]
            * self.reduce_block[1]
            * self.reduce_block[2],
        )

        # Process through adapter
        # x = x.reshape(-1, embedding_dim * self.reduce_block[0] * self.reduce_block[1] * self.reduce_block[2])
        x = self.ln_q(x)
        x = self.adapter(x)

        return x


class QwenVLModel3D(Qwen2_5_VLModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    config: Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config, reduce_block=[2, 2, 2]):
        super().__init__(config)

        self.reduce_block = reduce_block
        self.vision_adapter = VisionAdapterReduceBlocks(
            embedding_dim=864, embed_dim=2048, reduce_block=reduce_block
        )
        # self.vision_adapter = VisionAdapterHeterogeneous(embedding_dim=864, embed_dim=1536, reduce_block=reduce_block)

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

        if pixel_values_videos is not None:
            ##########
            # Only these 3 lines (and bellow two lines) are different from the original Qwen2_5_VLModel
            pixel_values_videos = pixel_values_videos.to(self.vision_adapter.adapter[0].weight.dtype)
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
                    # Since we use the time dimension as spatial dimension we have to apply the reduce factor already here since it is not done in the rope function below
                    video[:] = torch.floor_divide(video, torch.tensor(self.reduce_block, device=video.device, dtype=video.dtype))
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
    


class QwenVLForConditionalGeneration3D(Qwen2_5_VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = QwenVLModel3D(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()


