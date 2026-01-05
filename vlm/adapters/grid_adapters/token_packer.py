import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from vlm.adapters.grid_adapters.base import GridAdapter
from vlm.adapters.layers.attention import MHAttention
from vlm.adapters.layers.dense import MLP
from math import prod


class TokenPacker(GridAdapter):
    def __init__(self, head_dim: int, depth: int, **kwargs):
        super().__init__(**kwargs)

        self.q_proj = nn.Linear(self.vision_embed_dim, self.vision_embed_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.vision_embed_dim, self.vision_embed_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.vision_embed_dim, self.vision_embed_dim, bias=self.bias)

        self.q_norm = self.norm_layer_class(self.vision_embed_dim)
        self.k_norm = self.norm_layer_class(self.vision_embed_dim)
        self.v_norm = self.norm_layer_class(self.vision_embed_dim)

        self.attn = MHAttention(
            in_dim=self.vision_embed_dim,
            hidden_dim=self.vision_embed_dim,
            out_dim=self.vision_embed_dim,
            head_dim=head_dim,
            qkv_bias=self.bias,
            out_bias=False
        )

        self.pre_dense_norm = self.norm_layer_class(self.vision_embed_dim)
        self.dense_projector = MLP(in_features=self.vision_embed_dim, out_features=self.text_embed_dim, depth=depth)

    @staticmethod
    def divide_features(x: torch.Tensor, strides: tuple[int, int, int]) -> torch.Tensor:
        """
        In order to do efficient point-to-region attention, we fold the
        spatial dimensions into the batch dimension.

        Args:
            x (torch.Tensor):
            strides (tuple[int, int, int]):
        Returns:
            torch.Tensor: The reshaped input x.
        """
        sT, sH, sW = strides
        B, T, H, W, _ = x.shape
        return rearrange(
            x, "B (t sT) (h sH) (w sW) C -> (B t h w) (sT sH sW) C",  # t corresponds to T' in the
            sT=sT, sH=sH, sW=sW,                                              # syntax used below
        ).contiguous()

    def forward(self, x: torch.Tensor, adapter_args: dict):
        x = x.permute(0, 2, 3, 4, 1)
        batch_size, T, H, W, embed_dim = x.shape

        key = self.k_norm(self.k_proj(x))
        value = self.v_norm(self.v_proj(x))

        video_grid_thw = tuple(adapter_args["video_grid_thw"][0].tolist())
        query = F.interpolate(x.permute(0, 4, 1, 2, 3).float(), size=video_grid_thw,
                              mode="trilinear", align_corners=False)
        query = query.permute(0, 2, 3, 4, 1).to(x.dtype)
        query = self.q_norm(self.q_proj(query))

        # query.shape: (B, T', H', W', C)
        # key.shape:   (B,  T,  H,  W, C)
        # value.shape: (B,  T,  H,  W, C)

        reshaped_query = self.divide_features(query, strides=(1, 1, 1))
        reshaped_key = self.divide_features(key, strides=self.reduce_block)
        reshaped_value = self.divide_features(value, strides=self.reduce_block)

        # reshaped_query.shape: (B*T'*H'*W',         1, C)
        # reshaped_key.shape:   (B*T'*H'*W',  rT*rH*rW, C)
        # reshaped_value.shape: (B*T'*H'*W',  rT*rH*rW, C)

        x = self.attn(reshaped_query, reshaped_key, reshaped_value)

        num_queries = prod(video_grid_thw)
        x = x.reshape(batch_size, num_queries, -1)

        x = self.dense_projector(self.pre_dense_norm(x))
        return x


class TokenPackerNoProj(GridAdapter):
    def __init__(self, head_dim: int, depth: int, **kwargs):
        super().__init__(**kwargs)

        self.q_norm = self.norm_layer_class(self.vision_embed_dim)
        self.kv_norm = self.norm_layer_class(self.vision_embed_dim)

        self.attn = MHAttention(
            in_dim=self.vision_embed_dim,
            hidden_dim=self.vision_embed_dim,
            out_dim=self.vision_embed_dim,
            head_dim=head_dim,
            qkv_bias=self.bias,
            out_bias=False
        )

        self.pre_dense_norm = self.norm_layer_class(self.vision_embed_dim)
        self.dense_projector = MLP(in_features=self.vision_embed_dim, out_features=self.text_embed_dim, depth=depth)

    @staticmethod
    def divide_features(x: torch.Tensor, strides: tuple[int, int, int]) -> torch.Tensor:
        """
        In order to do efficient point-to-region attention, we fold the
        spatial dimensions into the batch dimension.

        Args:
            x (torch.Tensor):
            strides (tuple[int, int, int]):
        Returns:
            torch.Tensor: The reshaped input x.
        """
        sT, sH, sW = strides
        B, T, H, W, _ = x.shape
        return rearrange(
            x, "B (t sT) (h sH) (w sW) C -> (B t h w) (sT sH sW) C",  # t corresponds to T' in the
            sT=sT, sH=sH, sW=sW,                                              # syntax used below
        ).contiguous()

    def forward(self, x: torch.Tensor, adapter_args: dict):
        x = x.permute(0, 2, 3, 4, 1)
        batch_size, T, H, W, embed_dim = x.shape

        key = value = self.kv_norm(x)

        video_grid_thw = tuple(adapter_args["video_grid_thw"][0].tolist())
        query = F.interpolate(x.permute(0, 4, 1, 2, 3).float(), size=video_grid_thw,
                              mode="trilinear", align_corners=False)
        query = query.permute(0, 2, 3, 4, 1).to(x.dtype)
        query = self.q_norm(query)

        # query.shape: (B, T', H', W', C)
        # key.shape:   (B,  T,  H,  W, C)
        # value.shape: (B,  T,  H,  W, C)

        reshaped_query = self.divide_features(query, strides=(1, 1, 1))
        reshaped_key = self.divide_features(key, strides=self.reduce_block)
        reshaped_value = self.divide_features(value, strides=self.reduce_block)

        # reshaped_query.shape: (B*T'*H'*W',         1, C)
        # reshaped_key.shape:   (B*T'*H'*W',  rT*rH*rW, C)
        # reshaped_value.shape: (B*T'*H'*W',  rT*rH*rW, C)

        x = self.attn(reshaped_query, reshaped_key, reshaped_value)

        num_queries = prod(video_grid_thw)
        x = x.reshape(batch_size, num_queries, -1)

        x = self.dense_projector(self.pre_dense_norm(x))
        return x


class TokenPackerLLM(GridAdapter):
    def __init__(self, head_dim: int, depth: int, **kwargs):
        super().__init__(**kwargs)

        self.q_proj = nn.Linear(self.vision_embed_dim, self.vision_embed_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.vision_embed_dim, self.vision_embed_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.vision_embed_dim, self.vision_embed_dim, bias=self.bias)

        self.q_norm = self.norm_layer_class(self.vision_embed_dim)
        self.k_norm = self.norm_layer_class(self.vision_embed_dim)
        self.v_norm = self.norm_layer_class(self.vision_embed_dim)

        self.attn = MHAttention(
            in_dim=self.vision_embed_dim,
            hidden_dim=self.text_embed_dim,
            out_dim=self.text_embed_dim,
            head_dim=head_dim,
            qkv_bias=self.bias,
            out_bias=False
        )

        self.pre_dense_norm = self.norm_layer_class(self.text_embed_dim)
        self.dense_projector = MLP(in_features=self.text_embed_dim, out_features=self.text_embed_dim, depth=depth)

    @staticmethod
    def divide_features(x: torch.Tensor, strides: tuple[int, int, int]) -> torch.Tensor:
        """
        In order to do efficient point-to-region attention, we fold the
        spatial dimensions into the batch dimension.

        Args:
            x (torch.Tensor):
            strides (tuple[int, int, int]):
        Returns:
            torch.Tensor: The reshaped input x.
        """
        sT, sH, sW = strides
        B, T, H, W, _ = x.shape
        return rearrange(
            x, "B (t sT) (h sH) (w sW) C -> (B t h w) (sT sH sW) C",  # t corresponds to T' in the
            sT=sT, sH=sH, sW=sW,                                              # syntax used below
        )

    def forward(self, x: torch.Tensor, adapter_args: dict):
        x = x.permute(0, 2, 3, 4, 1)
        batch_size, T, H, W, embed_dim = x.shape

        key = self.k_norm(self.k_proj(x))
        value = self.v_norm(self.v_proj(x))

        video_grid_thw = tuple(adapter_args["video_grid_thw"][0].tolist())
        query = F.interpolate(x.permute(0, 4, 1, 2, 3).float(), size=video_grid_thw,
                              mode="trilinear", align_corners=False)
        query = query.permute(0, 2, 3, 4, 1).to(x.dtype)
        query = self.q_norm(self.q_proj(query))

        # query.shape: (B, T', H', W', C)
        # key.shape:   (B,  T,  H,  W, C)
        # value.shape: (B,  T,  H,  W, C)

        reshaped_query = self.divide_features(query, strides=(1, 1, 1))
        reshaped_key = self.divide_features(key, strides=self.reduce_block)
        reshaped_value = self.divide_features(value, strides=self.reduce_block)

        # reshaped_query.shape: (B*T'*H'*W',         1, C)
        # reshaped_key.shape:   (B*T'*H'*W',  rT*rH*rW, C)
        # reshaped_value.shape: (B*T'*H'*W',  rT*rH*rW, C)

        x = self.attn(reshaped_query, reshaped_key, reshaped_value)

        num_queries = prod(video_grid_thw)
        x = x.reshape(batch_size, num_queries, -1)

        x = self.dense_projector(self.pre_dense_norm(x))
        return x