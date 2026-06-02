from typing import Tuple

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.activations import QuickGELUActivation
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
from einops import rearrange
from vlm.adapters.grid_adapters.base import GridAdapter


class QwenAdapter(GridAdapter):
    def __init__(self, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        rT, rH, rW = self.reduce_block
        input_dim = self.vision_embed_dim * rT * rH * rW
        if hidden_dim is None:
            hidden_dim = self.vision_embed_dim * min(8, rT * rH * rW)
        self.rmsnorm = Qwen2RMSNorm(input_dim, eps=1e-6)
        self.adapter = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(hidden_dim, self.text_embed_dim)
        )

    def forward(self, x: torch.Tensor, adapter_args: dict):
        # Convert input to float first
        video_grid_thw = adapter_args['video_grid_thw']

        x = x.to(torch.bfloat16)

        # cur_pos = 0
        # TODO reshape input to expected length note that grid_thw is the grid
        # after the reduceblock in the upstream code...
        # TODO: also keep this in mind:
        # if x.ndim == 3:  # if flattened tensor (from VE, not loaded embed), restructure to original grid
        #     video_grid_thw = adapter_args["video_grid_thw"]
        #     x = x.reshape(batch_size, embedding_dim, *(video_grid_thw[0] * torch.tensor(self.reduce_block).to(video_grid_thw.device)))

        res = torch.Tensor().to(x.device, dtype=x.dtype)
        for image_idx, grid_thw in enumerate(video_grid_thw):
            # lenght = grid_thw[0] * grid_thw[1] * grid_thw[2]
            # y = x[cur_pos:cur_pos+lenght]
            y = x[image_idx]

            # (T*H*W, C) -> (C, T, H, W)
            # y = y.permute(1, 0)  # (T*H*W, C) -> (C, T*H*W)
            # rebuild blocks
            # y = y.view(y.shape[0], grid_thw[0], grid_thw[1], grid_thw[2])
            # Space-to-Depth rearrange (Depth as in Channel/hidden-dim, e.g. imagine a grid of cards which you now compress by stacking reduce block**3 cards onto each other)
            y = rearrange(
                y,
                # 'b c (t p1) (h p2) (w p3) -> b (t h w) (c p1 p2 p3)',
                'c (t p1) (h p2) (w p3) -> (t h w) (c p1 p2 p3)',
                p1=self.reduce_block[0],
                p2=self.reduce_block[1],
                p3=self.reduce_block[2]
            ).contiguous()
            res = torch.cat((res, y), dim=0)
            # cur_pos += lenght
        x = res

        # Process through adapter
        x = self.rmsnorm(x)
        x = self.adapter(x)

        return x


class QwenAdapterHeterogeneous(nn.Module):
    """
    Vision Adapter that handles heterogeneous input sequence lengths within a batch.

    Processes 3D feature maps (flattened into sequences) of varying sizes,
    reduces them using unfolding, maps them to the language model embedding
    space, and returns a padded batch with an attention mask.
    """

    def __init__(
        self,
        vision_embed_dim: int,
        text_embed_dim: int,
        reduce_block: tuple[int, int, int]
    ):
        """
        Args:
            vision_embed_dim (int): Dimension of the input vision embeddings (channel dim).
            text_embed_dim (int): Target embedding dimension (language model's embedding dim).
            reduce_block (Tuple[int, int, int]): Factors to reduce each spatial dimension (H, W, D) by.
                                                 Defines the size of the 3D patch to combine.
        """
        super().__init__()
        self.embedding_dim = vision_embed_dim
        self.embed_dim = text_embed_dim
        self.reduce_block = reduce_block
        self.reduce_factor = reduce_block[0] * reduce_block[1] * reduce_block[2]
        self.combined_input_dim = vision_embed_dim * self.reduce_factor

        # The adapter MLP
        self.adapter = nn.Sequential(
            nn.Linear(self.combined_input_dim, 5120),
            QuickGELUActivation(),
            nn.Linear(5120, text_embed_dim),
        )

    def forward(
        self, x: torch.Tensor, video_grid_thw: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
