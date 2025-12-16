import torch
import torch.nn as nn
from vlm.adapters.grid_adapters.base import GridAdapter
from vlm.adapters.layers.dense import MLP


class AveragePool(GridAdapter):
    def __init__(self, depth: int, **kwargs):
        super().__init__(**kwargs)
        self.avg_pool = nn.AvgPool3d(kernel_size=self.reduce_block, stride=self.reduce_block)
        self.norm = self.norm_layer_class(self.vision_embed_dim)
        self.dense_projector = MLP(self.vision_embed_dim, self.text_embed_dim, depth=depth)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        x = self.avg_pool(x)
        x = x.permute(0, 2, 3, 4, 1)
        return self.dense_projector(self.norm(x))


class MaxPool(GridAdapter):
    def __init__(self, depth: int, **kwargs):
        super().__init__(**kwargs)
        self.max_pool = nn.MaxPool3d(kernel_size=self.reduce_block, stride=self.reduce_block)
        self.norm = self.norm_layer_class(self.vision_embed_dim)
        self.dense_projector = MLP(self.vision_embed_dim, self.text_embed_dim, depth=depth)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        x = self.max_pool(x)
        x = x.permute(0, 2, 3, 4, 1)
        return self.dense_projector(self.norm(x))