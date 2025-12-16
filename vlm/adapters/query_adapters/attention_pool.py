import torch
import torch.nn as nn
from einops import repeat

from vlm.adapters.layers.attention import MHXAttention
from vlm.adapters.layers.dense import MLP
from vlm.adapters.query_adapters.base import QueryAdapter


class AttentionPool(QueryAdapter):
    def __init__(self, head_dim: int, depth: int, **kwargs):
        super().__init__(**kwargs)

        self.query_tokens = nn.Parameter(
            torch.randn(self.num_query_tokens, self.vision_embed_dim) / self.vision_embed_dim ** 0.5
        )

        self.query_norm = self.norm_layer_class(self.vision_embed_dim)
        self.vision_norm = self.norm_layer_class(self.vision_embed_dim)

        self.attn = MHXAttention(
            model_dim=self.vision_embed_dim,
            context_dim=self.vision_embed_dim,
            hidden_dim=self.vision_embed_dim,
            head_dim=head_dim,
            qkv_bias=self.bias,
            out_bias=False
        )
        self.pre_dense_norm = self.norm_layer_class(self.vision_embed_dim)
        self.dense_projector = MLP(self.vision_embed_dim, self.text_embed_dim, depth=depth, bias=self.bias)

    def forward(self, x, adapter_args: dict):
        batch_size, embed_dim, T, H, W = x.shape
        x = x.reshape(batch_size, embed_dim, -1).permute(0, 2, 1)

        query_tokens = repeat(self.query_tokens, "n d -> b n d", b=x.shape[0])

        x = self.attn(latents=self.query_norm(query_tokens), context=self.vision_norm(x))
        x = self.dense_projector(self.pre_dense_norm(x))

        return x