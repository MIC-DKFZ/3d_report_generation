import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from vlm.utils.tensor_ops import rearrange_many


class SeperateKVMHXAttention(nn.Module):
    def __init__(self, model_dim: int, context_dim: int, hidden_dim: int, head_dim: int,
                 qkv_bias: bool = True, out_bias: bool = False):
        super().__init__()
        assert hidden_dim % head_dim == 0, (f"'hidden_dim' must be divisible by 'head_dim': "
                                             f"hidden_dim={hidden_dim}, head_dim={head_dim}")

        self.to_q = nn.Linear(model_dim, hidden_dim, bias=qkv_bias)
        self.to_kv_1 = nn.Linear(context_dim, 2 * hidden_dim, bias=qkv_bias)
        self.to_kv_2 = nn.Linear(context_dim, 2 * hidden_dim, bias=qkv_bias)
        self.to_out = nn.Linear(hidden_dim, model_dim, bias=out_bias)

        self.num_heads = hidden_dim // head_dim

    def forward(self, latents: torch.Tensor, context_1: torch.Tensor, context_2: torch.Tensor):
        batch_size, num_latents, _ = latents.shape

        q = self.to_q(latents)
        k1, v1 = self.to_kv_1(context_1).chunk(2, dim=-1)
        k2, v2 = self.to_kv_2(context_2).chunk(2, dim=-1)
        k = torch.cat([k1, k2], dim=1)
        v = torch.cat([v1, v2], dim=1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        attn_out = scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_latents, -1)

        return self.to_out(attn_out)


class MHXAttention(nn.Module):
    def __init__(self, model_dim: int, context_dim: int, hidden_dim: int, head_dim: int,
                 qkv_bias: bool = True, out_bias: bool = False):
        super().__init__()
        assert hidden_dim % head_dim == 0, (f"'hidden_dim' must be divisible by 'head_dim': "
                                             f"hidden_dim={hidden_dim}, head_dim={head_dim}")

        self.to_q = nn.Linear(model_dim, hidden_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, 2 * hidden_dim, bias=qkv_bias)
        self.to_out = nn.Linear(hidden_dim, model_dim, bias=out_bias)

        self.num_heads = hidden_dim // head_dim

    def forward(self, latents: torch.Tensor, context: torch.Tensor):
        batch_size, num_latents, _ = latents.shape

        q = self.to_q(latents)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        attn_out = scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_latents, -1)

        return self.to_out(attn_out)


class MHAttention(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim, head_dim: int,
                 qkv_bias: bool = True, out_bias: bool = False):
        super().__init__()
        assert hidden_dim % head_dim == 0, (f"'hidden_dim' must be divisible by 'head_dim': "
                                             f"hidden_dim={hidden_dim}, head_dim={head_dim}")

        self.to_q = nn.Linear(in_dim, hidden_dim, bias=qkv_bias)
        self.to_k = nn.Linear(in_dim, hidden_dim, bias=qkv_bias)
        self.to_v = nn.Linear(in_dim, hidden_dim, bias=qkv_bias)
        self.to_out = nn.Linear(hidden_dim, out_dim, bias=out_bias)

        self.num_heads = hidden_dim // head_dim

    def forward(self, query, key, value):
        batch_size, num_queries, _ = query.shape

        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        attn_out = scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_queries, -1)

        return self.to_out(attn_out)
