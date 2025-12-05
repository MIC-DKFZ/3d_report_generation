import torch.nn as nn
from vlm.adapters.grid_adapters.base import GridAdapter
from vlm.adapters.layers.dense import MLP


class AveragePoolMLP(GridAdapter):
    def __init__(self, depth: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_pool = nn.AvgPool3d(kernel_size=self.reduce_block, stride=self.reduce_block)
        self.mlp = MLP(self.vision_embed_dim, self.text_embed_dim, depth=depth)

    def forward(self, x):
        return self.mlp(self.avg_pool(x))


class MaxPoolMLP(GridAdapter):
    def __init__(self, depth: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pool = nn.MaxPool3d(kernel_size=self.reduce_block, stride=self.reduce_block)
        self.mlp = MLP(self.vision_embed_dim, self.text_embed_dim, depth=depth)

    def forward(self, x):
        return self.mlp(self.avg_pool(x))