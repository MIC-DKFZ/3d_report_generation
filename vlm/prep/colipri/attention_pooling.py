import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

# from nnssl.online_probing.utils import cls_probe_aggregation_type


def get_sequence_pooling_class(pooling_type, pool_kwargs: dict) -> nn.Module:
    """
        Returns an sequence pooling module based on the specified pooling type.


    Args:
         pooling_type (cls_probe_aggregation_type): The type of pooling to use. Supported values are:
            - "AvgPool": Uses nn.AdaptiveAvgPool3d for average pooling.
            - "MaxPool": Uses nn.AdaptiveMaxPool3d for max pooling.
            - "LearnedAttentionPool": Uses LearnedQueryAttentionPool1D for learned attention pooling.
            - "AvgAttentionPool": Uses AttentionPool1D for average attention pooling.
            - "MaxAttentionPool": Uses MultiLearnedQueryAttentionPool1D for max attention pooling.
        pool_kwargs (dict): Additional keyword arguments to pass to the attention pooling modules.

    Raises:
        ValueError: If an unknown pooling type is provided.

    Returns:
        nn.Module:  nn.Module: An instance of the selected pooling module.
    """
    if pooling_type == "AvgPool":
        global_pooling_module = nn.Sequential(Rearrange("n t d -> n d t"), nn.AdaptiveAvgPool1d(1))
    elif pooling_type == "MaxPool":
        global_pooling_module = nn.Sequential(Rearrange("n t d -> n d t"), nn.AdaptiveMaxPool1d(1))
    elif pooling_type == "LearnedAttentionPool":
        global_pooling_module = LearnedQueryAttentionPool1D(**pool_kwargs)
    elif pooling_type == "AvgAttentionPool":
        global_pooling_module = AttentionPool1D(**pool_kwargs)
    elif pooling_type == "MultiLearnedAttentionPool":
        global_pooling_module = MultiLearnedQueryAttentionPool1D(**pool_kwargs)
    elif pooling_type == "LearnedQueryAttentionMaxPool1D":
        global_pooling_module = LearnedQueryAttentionMaxPool1D(**pool_kwargs)
    else:
        raise ValueError(f"Unknown pooling type {pooling_type}")
    return global_pooling_module


class AttentionPool1D(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.num_heads = num_heads

    def forward(self, x):
        """
        x: [B, T, D] — sequence of token embeddings
        returns: [B, D] — pooled representation
        """
        query = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        pooled, _ = self.attn(query, x, x)  # [B, 1, D]
        return pooled.squeeze(1)  # [B, D]


class LastTokenPool1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: [B, T, D] — sequence of token embeddings
        returns: [B, D] — pooled representation
        """
        return x[:, -1, :]  # [B, D]

class LearnedQueryAttentionPool1D(AttentionPool1D):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__(embed_dim, num_heads)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) / embed_dim**0.5)

    def forward(self, x):
        """
        x: [B, T, D] — sequence of token embeddings
        returns: [B, D] — pooled representation
        """
        B, T, D = x.shape
        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        pooled, _ = self.attn(query, x, x)  # [B, 1, D]
        return pooled.squeeze(dim=1)  # [B, D]


class MultiLearnedQueryAttentionPool1D(AttentionPool1D):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__(embed_dim, num_heads)
        # 4 Queries instead of 1
        self.query = nn.Parameter(torch.randn(1, 4, embed_dim) / embed_dim**0.5)

    def forward(self, x):
        """
        x: [B, T, D] — sequence of token embeddings
        returns: [B, D] — pooled representation
        """
        B, T, D = x.shape
        query = self.query.expand(B, -1, -1)  # [B, 4, D]
        pooled, _ = self.attn(query, x, x)  # [B, 4, D]
        # pooled: [4, B, D_out], want [B, D_out] by pooling over queries (mean)
        return pooled.mean(dim=1)  # [B, D]


class LearnedQueryAttentionMaxPool1D(AttentionPool1D):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__(embed_dim, num_heads)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) / embed_dim**0.5)
        # Try for torch.compile
        self.token_max = nn.Sequential(
            Rearrange("n t d -> n d t"),
            nn.AdaptiveMaxPool1d(1),
            Rearrange("n d t -> n t d"),
        )

    def forward(self, x):
        """
        x: [B, T, D] — sequence of token embeddings
        returns: [B, D] — pooled representation
        """
        B, T, D = x.shape
        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        query_offset = self.token_max(x)  # [B, 1, D]
        query = query + query_offset
        pooled, _ = self.attn(query, x, x)  # [B, 1, D]
        return pooled.squeeze(dim=1)  # [B, D]


class MaxPoolQueryAttention(AttentionPool1D):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__(embed_dim, num_heads)
        # Try for torch.compile
        self.token_max = nn.Sequential(
            Rearrange("n t d -> n d t"),
            nn.AdaptiveMaxPool1d(1),
            Rearrange("n d t -> n t d"),
        )

    def forward(self, x):
        """
        x: [B, T, D] — sequence of token embeddings
        returns: [B, D] — pooled representation
        """
        B, T, D = x.shape
        query = self.token_max(x)  # [B, 1, D]
        pooled, _ = self.attn(query, x, x)  # [B, 1, D]
        return pooled.squeeze(dim=1)  # [B, D]
