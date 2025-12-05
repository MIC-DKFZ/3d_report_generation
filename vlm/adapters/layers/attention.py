import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

from vlm.utils.tensor_ops import rearrange_many


class MHXAttention(nn.Module):
    def __init__(self, model_dim: int, context_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, (f"'hidden_dim' must be divisible by 'num_heads': "
                                             f"hidden_dim={hidden_dim}, num_heads={num_heads}")
        self.latents_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(model_dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, 2 * hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, model_dim)

        self.num_heads = num_heads


    def forward(self, latents, context):
        batch_size, num_latents, _ = latents.shape

        latents = self.latents_norm(latents)
        context = self.context_norm(context)

        q = self.to_q(latents)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        attn_out = scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_latents, -1)

        return self.to_out(attn_out)