from math import prod
import torch
from einops import rearrange

from vlm.models.adapters.grid_adapters.base import GridAdapter
from vlm.models.adapters.layers.dense import MLP


class ConcatAdapter(GridAdapter):
    def __init__(self, depth: int, **kwargs):
        super().__init__(**kwargs)
        input_dim = self.vision_embed_dim * prod(self.reduce_block)
        self.norm = self.norm_layer_class(input_dim)

        bottleneck_dims = [
            self._get_clamped_dim(power=2, max_mult=3),
            self._get_clamped_dim(power=1, max_mult=1.5)
        ]

        # 3. Construct Hidden Layers based on Depth
        # Take up to the first 2 calculated bottlenecks, fill the rest with text_embed_dim
        hidden_dims = bottleneck_dims[:depth] + \
                      [self.text_embed_dim] * max(0, depth - 2)

        # 4. Final Projection
        all_dims = [input_dim, *hidden_dims, self.text_embed_dim]
        self.dense_projector = MLP(features=all_dims, bias=self.bias)

    def _get_clamped_dim(self, power: int, max_mult: float, multiple: int = 32) -> int:
        """
        Calculates a dimension size derived from the 'reduce_block factor.
        The result is rounded up to a multiple of 'multiple' and clamped between
        text_embed_dim and (max_mult * text_embed_dim).
        """
        raw_dim = self.vision_embed_dim * (self.reduce_block[-1] ** power)
        rounded_raw_dim = ((raw_dim + multiple - 1) // multiple) * multiple

        lower_bound = self.text_embed_dim
        upper_bound = int(max_mult * self.text_embed_dim)

        return max(lower_bound, min(rounded_raw_dim, upper_bound))

    def forward(self, x: torch.Tensor, adapter_args: dict):
        # batch_size, embed_dim, T, H, W = x.shape

        rT, rH, rW = self.reduce_block
        x = rearrange(x, "B C (t rT) (h rH) (w rW) -> B (t h w) (rT rH rW C)", rT=rT, rH=rH, rW=rW).contiguous()

        # Process through adapter
        x = self.norm(x)
        x = self.dense_projector(x)

        return x
