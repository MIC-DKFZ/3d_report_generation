from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset

from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

from transformers import BitsAndBytesConfig
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from transformers.activations import QuickGELUActivation

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast, QWEN2_VL_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionMlp
from transformers.models.qwen2_vl.modeling_qwen2_vl import ACT2FN

from qwen_vl_utils import process_vision_info


'''

    def __init__(self, embedding_dim: int, embed_dim: int):
        super().__init__()
        self.adapter = VisionMlp(
            dim=embedding_dim,  # 864
            hidden_dim=embed_dim,  # 1536
            hidden_act="quick_gelu"
        )
'''

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
        x = x.reshape(batch_size * seq_len, -1)  # -> (batch_size * sequence_length, 1536)

        return x


class VisionAdapterReduceBlocks(nn.Module):
    def __init__(self, embedding_dim: int, embed_dim: int, reduce_block: Tuple[int, int, int] = [2, 2, 2]):
        super().__init__()
        self.reduce_block = reduce_block
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim * reduce_block[0] * reduce_block[1] * reduce_block[2], 5120),
            QuickGELUActivation(),
            nn.Linear(5120, embed_dim),
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
        x = x.reshape(batch_size, -1, embedding_dim * self.reduce_block[0] * self.reduce_block[1] * self.reduce_block[2])

        # Process through adapter
        # x = x.reshape(-1, embedding_dim * self.reduce_block[0] * self.reduce_block[1] * self.reduce_block[2])
        x = self.adapter(x)

        return x


class VisionAdapterHeterogeneous(nn.Module):
    """
    Vision Adapter that handles heterogeneous input sequence lengths within a batch.

    Processes 3D feature maps (flattened into sequences) of varying sizes,
    reduces them using unfolding, maps them to the language model embedding
    space, and returns a padded batch with an attention mask.
    """
    def __init__(self, embedding_dim: int, embed_dim: int, reduce_block: Tuple[int, int, int] = (2, 2, 2)):
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

    def forward(self, x: torch.Tensor, video_grid_thw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            grid_thw = video_grid_thw[i].tolist() # List [T_i, H_i, W_i]
            seq_length_i = x_item.shape[1]

            # --- Sanity Check ---
            expected_seq_length = grid_thw[0] * grid_thw[1] * grid_thw[2]
            if seq_length_i != expected_seq_length:
                raise ValueError(f"Batch item {i}: seq_length {seq_length_i} does not match grid {grid_thw} product {expected_seq_length}")

            # --- 2. Isolate & Reshape to 5D ---
            # Reshape to (1, C, T, H, W) for unfold compatibility
            try:
                x_item_5d = x_item.reshape(1, self.embedding_dim, *grid_thw)
            except RuntimeError as e:
                 raise RuntimeError(f"Error reshaping batch item {i} with embedding_dim {self.embedding_dim}, seq_len {seq_length_i} to grid {grid_thw}. Error: {e}")

            # --- 3. Pad Spatially if Dimensions are not multiples of reduce_block ---
            current_t, current_h, current_w = grid_thw
            rb_t, rb_h, rb_w = self.reduce_block

            # Calculate padding needed for T, H, W dimensions
            # If dimension is already a multiple, padding is 0.
            # Otherwise, calculate padding to reach the next multiple.
            pad_t = (rb_t - current_t % rb_t) % rb_t
            pad_h = (rb_h - current_h % rb_h) % rb_h
            pad_w = (rb_w - current_w % rb_w) % rb_w

            x_item_5d_padded = x_item_5d # Start assuming no padding needed
            if pad_t > 0 or pad_h > 0 or pad_w > 0:
                # PyTorch's F.pad expects padding amounts in reverse order of dimensions:
                # (pad_last_dim_left, pad_last_dim_right, pad_penultimate_dim_top, pad_penultimate_dim_bottom, ...)
                # For (N, C, T, H, W), the order is (W_left, W_right, H_top, H_bottom, T_front, T_back)
                # We pad only at the 'end' (right/bottom/back)
                paddings = (0, pad_w, 0, pad_h, 0, pad_t)
                x_item_5d_padded = F.pad(x_item_5d, paddings, mode='constant', value=0.0)
                # print(f"Item {i}: Original grid {grid_thw}, Padding applied ({pad_t}, {pad_h}, {pad_w}), New shape {x_item_5d_padded.shape}") # Optional debug print

            # Store the new, potentially padded dimensions
            padded_grid_thw = [current_t + pad_t, current_h + pad_h, current_w + pad_w]

            # Calculate the output grid size after reduction (based on padded dimensions)
            # Division will now be exact.
            out_grid = [g // r for g, r in zip(padded_grid_thw, self.reduce_block)]



            # --- 4. Unfold & Reduce ---
            try:
                # Unfold along T, H, W dimensions
                unfolded_x = x_item_5d.unfold(2, self.reduce_block[0], self.reduce_block[0]) # Dim T
                unfolded_x = unfolded_x.unfold(3, self.reduce_block[1], self.reduce_block[1]) # Dim H
                unfolded_x = unfolded_x.unfold(4, self.reduce_block[2], self.reduce_block[2]) # Dim W
                # Current shape: (1, C, T', H', W', rT, rH, rW)
                # where T' = T / rT (or floor), etc.

                # Permute to group reduction dims with channel dim: (1, T', H', W', C, rT, rH, rW)
                permuted_x = unfolded_x.permute(0, 2, 3, 4, 1, 5, 6, 7)

                # Reshape to (1, T'*H'*W', C*rT*rH*rW) = (1, num_reduced_tokens_i, combined_input_dim)
                num_reduced_tokens_i = out_grid[0] * out_grid[1] * out_grid[2]
                # Use view or reshape. View is generally preferred if possible.
                # We calculated combined_input_dim in __init__
                reduced_x = permuted_x.reshape(1, num_reduced_tokens_i, self.combined_input_dim)

                # Remove the batch dimension (was 1 anyway): (num_reduced_tokens_i, combined_input_dim)
                reduced_x = reduced_x.squeeze(0)

            except RuntimeError as e:
                raise RuntimeError(f"Error during unfold/reshape for batch item {i} with grid {grid_thw}, reduce_block {self.reduce_block}. Shape after reshape: {x_item_5d.shape}. Error: {e}")

            # --- 5. Adapt ---
            # Pass through the adapter
            mapped_x = self.adapter(reduced_x) # Shape: (num_reduced_tokens_i, embed_dim)

            # --- 6. Collect ---
            processed_sequences.append(mapped_x)
            output_lengths.append(mapped_x.shape[0]) # Store the length *after* reduction

        # --- 6. Pad & Mask ---
        # Use pad_sequence to handle padding. It expects a list of tensors (L_i, *dims).
        # batch_first=True gives output shape (B, L_max, *dims)
        padded_output = pad_sequence(processed_sequences, batch_first=True, padding_value=0.0)
        # Shape: (batch_size, max_reduced_len, embed_dim)

        # Create the attention mask
        max_len = padded_output.shape[1]
        # Create a tensor of indices [0, 1, ..., max_len-1]
        indices = torch.arange(max_len, device=padded_output.device)
        # Create the mask by comparing indices with actual lengths
        # Shape: (batch_size, max_len). True for valid tokens, False for padding.
        attention_mask = indices[None, :] < torch.tensor(output_lengths, device=padded_output.device)[:, None]

        return padded_output, attention_mask


from torch.nn import CrossEntropyLoss
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings

class Qwen2VLForConditionalGeneration3D(Qwen2VLForConditionalGeneration):
    def __init__(self, config, reduce_block = [2, 2, 2]):
            super().__init__(config)

            self.reduce_block = reduce_block
            self.vision_adapter = VisionAdapterReduceBlocks(embedding_dim=864, embed_dim=1536, reduce_block=reduce_block)
            # self.vision_adapter = VisionAdapterHeterogeneous(embedding_dim=864, embed_dim=1536, reduce_block=reduce_block)

    @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
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
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                #pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                #video_embeds = self.vision_adapter(pixel_values_videos) # The only line that has changed so far (and the added adapter above)
                pixel_values_videos = pixel_values_videos.to(self.vision_adapter.adapter[0].weight.dtype)
                video_embeds = self.vision_adapter(pixel_values_videos, video_grid_thw)

                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0] * video_embeds.shape[1]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                for video in video_grid_thw:
                    # Since we use the time dimension as spatial dimension we have to apply the reduce factor already here since it is not done in the rope function below
                    video[0] = torch.floor_divide(video[0], self.reduce_block[0])
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
