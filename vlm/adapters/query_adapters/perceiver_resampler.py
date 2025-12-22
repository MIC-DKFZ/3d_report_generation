import torch
import torch.nn as nn
from einops import repeat

from vlm.adapters.layers.attention import MHXAttention
from vlm.adapters.layers.dense import MLP
from vlm.adapters.query_adapters.base import QueryAdapter

class PerceiverBlock(nn.Module):
    def __init__(self, model_dim: int, context_dim: int, head_dim: int, bias: bool, norm_layer_class: type):
        super().__init__()
        self.latents_norm = norm_layer_class(model_dim)
        self.context_norm = norm_layer_class(context_dim)
        self.attn = MHXAttention(
            model_dim=model_dim,
            context_dim=context_dim,
            hidden_dim=model_dim,
            head_dim=head_dim,
            qkv_bias=bias,
            out_bias=False
        )
        self.pre_ffw_norm = norm_layer_class(model_dim)
        self.ffw = MLP(features=[model_dim, model_dim, model_dim], bias=bias)

    def forward(self, latents: torch.Tensor, context: torch.Tensor):
        latents = self.latents_norm(latents)
        context = self.context_norm(context)

        context = torch.concat([context, latents], dim=-2)

        latents = latents + self.attn(latents, context)
        latents = latents + self.ffw(self.pre_ffw_norm(latents))

        return latents


class PerceiverResampler(QueryAdapter):
    def __init__(self, head_dim: int, depth: int, **kwargs):
        super().__init__(**kwargs)

        self.query_tokens = nn.Parameter(
            torch.randn(self.num_query_tokens, self.vision_embed_dim) / self.vision_embed_dim ** 0.5
        )
        self.blocks = nn.ModuleList([
            PerceiverBlock(self.vision_embed_dim, self.vision_embed_dim, head_dim, self.bias, self.norm_layer_class) for _ in range(depth)
        ])
        self.pre_dense_norm = self.norm_layer_class(self.vision_embed_dim)
        self.dense_projector = nn.Linear(self.vision_embed_dim, self.text_embed_dim, bias=True)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        batch_size, embed_dim, T, H, W = x.shape
        x = x.reshape(batch_size, embed_dim, -1).permute(0, 2, 1)

        vision_tokens = x
        latents = repeat(self.query_tokens, "n d -> b n d", b=batch_size)

        for perceiver_block in self.blocks:
            latents = perceiver_block(latents, vision_tokens)

        # normalize first due to skip-connection, then expand embed dim to text embed dim
        latents = self.dense_projector(self.pre_dense_norm(latents))

        return latents


class PerceiverResamplerLLM(QueryAdapter):
    def __init__(self, head_dim: int, depth: int, **kwargs):
        super().__init__(**kwargs)

        self.query_tokens = nn.Parameter(
            torch.randn(self.num_query_tokens, self.text_embed_dim) / self.text_embed_dim ** 0.5
        )
        self.vision_proj = nn.Linear(self.vision_embed_dim, self.text_embed_dim)
        self.blocks = nn.ModuleList([
            PerceiverBlock(self.text_embed_dim, self.text_embed_dim, head_dim, self.bias, self.norm_layer_class) for _ in range(depth)
        ])
        self.pre_dense_norm = self.norm_layer_class(self.text_embed_dim)
        self.dense_projector = nn.Linear(self.text_embed_dim, self.text_embed_dim, bias=True)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        batch_size, embed_dim, T, H, W = x.shape
        x = x.reshape(batch_size, embed_dim, -1).permute(0, 2, 1)

        vision_tokens = self.vision_proj(x)
        latents = repeat(self.query_tokens, "n d -> b n d", b=batch_size)

        for perceiver_block in self.blocks:
            latents = perceiver_block(latents, vision_tokens)

        # normalize first due to skip-connection, then expand embed dim to text embed dim
        latents = self.dense_projector(self.pre_dense_norm(latents))

        return latents