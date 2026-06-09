import torch.nn as nn


_eps = 1e-5

class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=_eps, **kwargs):
        super().__init__(normalized_shape, eps=eps, **kwargs)


class RMSNorm(nn.RMSNorm):
    def __init__(self, normalized_shape, eps=_eps, **kwargs):
        super().__init__(normalized_shape, eps=eps, **kwargs)


NORM_LAYER_REGISTRY: list[type] = [
    LayerNorm,
    RMSNorm
]