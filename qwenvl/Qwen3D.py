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
from transformers.generation import GenerationMixin
from  transformers.utils import auto_docstring, can_return_tuple, TransformersKwargs, is_torchdynamo_compiling, logging

from prep.evavit import PretrainedEvaEncFromEvaMAE, get_config_from_plans

from einops import rearrange

class VisionAdapterHeterogeneous(nn.Module):
    """
    Vision Adapter that handles heterogeneous input sequence lengths within a batch.

    Processes 3D feature maps (flattened into sequences) of varying sizes,
    reduces them using unfolding, maps them to the language model embedding
    space, and returns a padded batch with an attention mask.
    """

    def __init__(
        self,
        embedding_dim: int,
        embed_dim: int,
        reduce_block: Tuple[int, int, int] = (2, 2, 2),
    ):
        """
        Args:
            embedding_dim (int): Dimension of the input vision embeddings (channel dim).
            embed_dim (int): Target embedding dimension (language model's embedding dim).
            reduce_block (Tuple[int, int, int]): Factors to reduce each spatial dimension (H, W, D) by.
                                                 Defines the size of the 3D patch to combine.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embed_dim = embed_dim
        self.reduce_block = reduce_block
        self.reduce_factor = reduce_block[0] * reduce_block[1] * reduce_block[2]
        self.combined_input_dim = embedding_dim * self.reduce_factor

        # The adapter MLP
        self.adapter = nn.Sequential(
            nn.Linear(self.combined_input_dim, 5120),
            QuickGELUActivation(),
            nn.Linear(5120, embed_dim),
        )

    def forward(
        self, x: torch.Tensor, video_grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for heterogeneous batch.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim, seq_length).
                              seq_length can vary across the batch.
            video_grid_thw (torch.Tensor): Tensor of shape (batch_size, 3) containing the
                                           original [T, H, W] dimensions for each item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - padded_output (torch.Tensor): The processed embeddings, padded to the maximum
                                                reduced sequence length in the batch.
                                                Shape: (batch_size, max_reduced_len, embed_dim).
                - attention_mask (torch.Tensor): Boolean mask indicating valid (True) vs padded (False)
                                                 tokens. Shape: (batch_size, max_reduced_len).
        """
        # Ensure input is float (bfloat16 used here as in original code)
        x = x.to(torch.bfloat16)
        batch_size = x.shape[0]

        processed_sequences = []
        output_lengths = []

        # --- 1. Iterate through batch ---
        for i in range(batch_size):
            x_item = x[i]  # Shape: (embedding_dim, seq_length_i)
            grid_thw = video_grid_thw[i].tolist()  # List [T_i, H_i, W_i]
            seq_length_i = x_item.shape[1]

            # --- Sanity Check ---
            expected_seq_length = grid_thw[0] * grid_thw[1] * grid_thw[2]
            if seq_length_i != expected_seq_length:
                raise ValueError(
                    f"Batch item {i}: seq_length {seq_length_i} does not match grid {grid_thw} product {expected_seq_length}"
                )

            # --- 2. Isolate & Reshape to 5D ---
            # Reshape to (1, C, T, H, W) for unfold compatibility
            try:
                x_item_5d = x_item.reshape(1, self.embedding_dim, *grid_thw)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Error reshaping batch item {i} with embedding_dim {self.embedding_dim}, seq_len {seq_length_i} to grid {grid_thw}. Error: {e}"
                )

            # --- 3. Pad Spatially if Dimensions are not multiples of reduce_block ---
            current_t, current_h, current_w = grid_thw
            rb_t, rb_h, rb_w = self.reduce_block

            # Calculate padding needed for T, H, W dimensions
            # If dimension is already a multiple, padding is 0.
            # Otherwise, calculate padding to reach the next multiple.
            pad_t = (rb_t - current_t % rb_t) % rb_t
            pad_h = (rb_h - current_h % rb_h) % rb_h
            pad_w = (rb_w - current_w % rb_w) % rb_w

            x_item_5d_padded = x_item_5d  # Start assuming no padding needed
            if pad_t > 0 or pad_h > 0 or pad_w > 0:
                # PyTorch's F.pad expects padding amounts in reverse order of dimensions:
                # (pad_last_dim_left, pad_last_dim_right, pad_penultimate_dim_top, pad_penultimate_dim_bottom, ...)
                # For (N, C, T, H, W), the order is (W_left, W_right, H_top, H_bottom, T_front, T_back)
                # We pad only at the 'end' (right/bottom/back)
                paddings = (0, pad_w, 0, pad_h, 0, pad_t)
                x_item_5d_padded = F.pad(
                    x_item_5d, paddings, mode="constant", value=0.0
                )
                # print(f"Item {i}: Original grid {grid_thw}, Padding applied ({pad_t}, {pad_h}, {pad_w}), New shape {x_item_5d_padded.shape}") # Optional debug print

            # Store the new, potentially padded dimensions
            padded_grid_thw = [current_t + pad_t, current_h + pad_h, current_w + pad_w]

            # Calculate the output grid size after reduction (based on padded dimensions)
            # Division will now be exact.
            out_grid = [g // r for g, r in zip(padded_grid_thw, self.reduce_block)]

            # --- 4. Unfold & Reduce ---
            try:
                # Unfold along T, H, W dimensions
                unfolded_x = x_item_5d.unfold(
                    2, self.reduce_block[0], self.reduce_block[0]
                )  # Dim T
                unfolded_x = unfolded_x.unfold(
                    3, self.reduce_block[1], self.reduce_block[1]
                )  # Dim H
                unfolded_x = unfolded_x.unfold(
                    4, self.reduce_block[2], self.reduce_block[2]
                )  # Dim W
                # Current shape: (1, C, T', H', W', rT, rH, rW)
                # where T' = T / rT (or floor), etc.

                # Permute to group reduction dims with channel dim: (1, T', H', W', C, rT, rH, rW)
                permuted_x = unfolded_x.permute(0, 2, 3, 4, 1, 5, 6, 7)

                # Reshape to (1, T'*H'*W', C*rT*rH*rW) = (1, num_reduced_tokens_i, combined_input_dim)
                num_reduced_tokens_i = out_grid[0] * out_grid[1] * out_grid[2]
                # Use view or reshape. View is generally preferred if possible.
                # We calculated combined_input_dim in __init__
                reduced_x = permuted_x.reshape(
                    1, num_reduced_tokens_i, self.combined_input_dim
                )

                # Remove the batch dimension (was 1 anyway): (num_reduced_tokens_i, combined_input_dim)
                reduced_x = reduced_x.squeeze(0)

            except RuntimeError as e:
                raise RuntimeError(
                    f"Error during unfold/reshape for batch item {i} with grid {grid_thw}, reduce_block {self.reduce_block}. Shape after reshape: {x_item_5d.shape}. Error: {e}"
                )

            # --- 5. Adapt ---
            # Pass through the adapter
            mapped_x = self.adapter(
                reduced_x
            )  # Shape: (num_reduced_tokens_i, embed_dim)

            # --- 6. Collect ---
            processed_sequences.append(mapped_x)
            output_lengths.append(
                mapped_x.shape[0]
            )  # Store the length *after* reduction

        # --- 6. Pad & Mask ---
        # Use pad_sequence to handle padding. It expects a list of tensors (L_i, *dims).
        # batch_first=True gives output shape (B, L_max, *dims)
        padded_output = pad_sequence(
            processed_sequences, batch_first=True, padding_value=0.0
        )
        # Shape: (batch_size, max_reduced_len, embed_dim)

        # Create the attention mask
        max_len = padded_output.shape[1]
        # Create a tensor of indices [0, 1, ..., max_len-1]
        indices = torch.arange(max_len, device=padded_output.device)
        # Create the mask by comparing indices with actual lengths
        # Shape: (batch_size, max_len). True for valid tokens, False for padding.
        attention_mask = (
            indices[None, :]
            < torch.tensor(output_lengths, device=padded_output.device)[:, None]
        )

        return padded_output, attention_mask


class VisionAdapterReduceBlocks(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        embed_dim: int,
        reduce_block: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        self.reduce_block = reduce_block
        print(reduce_block)
        input_size = embedding_dim * reduce_block[0] * reduce_block[1] * reduce_block[2]
        hidden_size = embedding_dim * min(8, reduce_block[0] * reduce_block[1] * reduce_block[2]) # without min, this quickly becomes too large for e.g. 8cube. Original Qwen25VL uses blocksize**2=4
        # the linear layer still can become too large for large blocks greater than 8cube. In such cases better use pooling?!
        self.ln_q = Qwen2RMSNorm(input_size, eps=1e-6)
        self.adapter = nn.Sequential(
            nn.Linear(
                input_size,
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

    def __init__(self, config, reduce_block=[2, 2, 2], train_vision_encoder=False, vision_encoder_path=None):
        super().__init__(config)

        self.reduce_block = reduce_block
        # Added by us
        self.vision_adapter = VisionAdapterReduceBlocks(
            embedding_dim=864, embed_dim=2048, reduce_block=reduce_block
        )

        self.train_vision_encoder = train_vision_encoder
        if self.train_vision_encoder:
            self.setup_vision_encoder(vision_encoder_path)

        # self.vision_adapter = VisionAdapterHeterogeneous(embedding_dim=864, embed_dim=1536, reduce_block=reduce_block)

    def setup_vision_encoder(self, base_path):
        plans_file = f"{base_path}/plans.json"
        checkpoint_path = f"{base_path}/fold_all/checkpoint_final.pth"

        config = get_config_from_plans(plans_file)
        self.vision_encoder = PretrainedEvaEncFromEvaMAE(
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
            pixel_values_videos = pixel_values_videos.to(self.vision_adapter.adapter[0].weight.dtype)
            if self.train_vision_encoder:
                video_embeds = self.vision_encoder(pixel_values_videos.unsqueeze(1))
                video_embeds = rearrange(video_embeds, 'b c h w d -> b c (h w d )')
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
                    # Since we use the time dimension as spatial dimension we have to apply the reduce factor already here since it is not done in the rope function below
                    # For reduceblocks different to 2,2,2 we have to also adjus dim 1 and 2
                    video[0] = torch.floor_divide(video[0], torch.tensor(self.reduce_block[0], device=video.device, dtype=video.dtype))
                    video[1] = torch.floor_divide(video[1], torch.tensor(self.reduce_block[1] // 2, device=video.device, dtype=video.dtype))
                    video[2] = torch.floor_divide(video[2], torch.tensor(self.reduce_block[2] // 2, device=video.device, dtype=video.dtype))
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
        r"^model(?!\.(language_model|visual|vision_adapter))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, reduce_block=[2, 2, 2], train_vision_encoder=False, vision_encoder_path=None):
        super().__init__(config)
        self.reduce_block = reduce_block
        self.model = QwenVLModel3D(config, reduce_block=self.reduce_block, train_vision_encoder=train_vision_encoder, vision_encoder_path=vision_encoder_path)
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

