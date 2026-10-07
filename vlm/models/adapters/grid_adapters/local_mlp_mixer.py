from math import prod

import torch
import torch.nn as nn
from einops import rearrange

from vlm.models.adapters.grid_adapters.base import GridAdapter
from vlm.models.adapters.layers.dense import MixerBlockNoSkips


class LocalMLPMixer(GridAdapter):
    """
    A 3D-aware MLP-Mixer adapter that applies spatial and channel mixing to local grid patches.

    This architecture chains `MixerBlock` layers to perform progressive spatial pooling (token reduction)
    and feature projection. It functions similarly to a depthwise-separable convolution but uses
    MLPs with non-linearities for greater expressivity.

    Args:
        tokens_and_features (list[tuple[int, int]]): The transformation path. Each tuple
            `(num_tokens, embed_dim)` defines the input/output shape of a block.
            Example: `[(8, 1056), (1, 2048)]` creates one block reducing 8 tokens to 1
            and projecting dimension 1056 to 2048.
        hidden_dim_factor (int): Expansion factor for the internal MLP hidden layers.
        **kwargs: Arguments for GridAdapter (reduce_block, norm_layer_class, etc.).
    """

    def __init__(self, depth: int, hidden_dim_factor: int = 4, **kwargs):
        super().__init__(**kwargs)

        lis: list[nn.Module] = []
        tokens_and_features: list[tuple[int, int]]

        num_local_tokens = prod(self.reduce_block)
        tokens_and_features = [(num_local_tokens, self.vision_embed_dim)]

        for _ in range(depth - 1):
            tokens_and_features.append((num_local_tokens, self.vision_embed_dim))

        tokens_and_features.append((1, self.text_embed_dim))

        for i in range(len(tokens_and_features) - 1):
            in_vision_tokens, in_features = tokens_and_features[i]
            out_vision_tokens, out_features = tokens_and_features[i + 1]
            lis.append(
                MixerBlockNoSkips(in_features, out_features, in_vision_tokens, out_vision_tokens,
                                  hidden_dim_factor, self.norm_layer_class, self.bias)
            )
        self.mixer_blocks = nn.Sequential(*lis)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        batch_size, embed_dim, T, H, W = x.shape

        rT, rH, rW = self.reduce_block
        x = rearrange(x, "B C (t rT) (h rH) (w rW) -> (B t h w) (rT rH rW) C", rT=rT, rH=rH, rW=rW).contiguous()

        x = self.mixer_blocks(x)
        x = rearrange(x, "(B N) 1 C -> B N C", B=batch_size)

        return x
