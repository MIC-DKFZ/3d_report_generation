from torch.nn import LayerNorm, RMSNorm


NORM_LAYER_REGISTRY: list[type] = [
    LayerNorm,
    RMSNorm
]