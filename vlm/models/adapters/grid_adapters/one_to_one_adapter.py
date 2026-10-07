import torch
from einops import rearrange

from vlm.models.adapters.grid_adapters.base import GridAdapter
from vlm.models.adapters.layers.dense import MLP


class OneToOneAdapter(GridAdapter):
    def __init__(self, depth: int, reduce_block: tuple[int, int, int] = (1, 1, 1), **kwargs):
        super().__init__(reduce_block=(1, 1, 1), **kwargs)
        input_dim = self.vision_embed_dim
        self.norm = self.norm_layer_class(input_dim)

        self.dense_projector = MLP(in_features=self.vision_embed_dim, out_features=self.text_embed_dim, depth=depth)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        # batch_size, embed_dim, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> b (t h w) c").contiguous()

        # Process through adapter
        x = self.norm(x)
        x = self.dense_projector(x)

        return x
