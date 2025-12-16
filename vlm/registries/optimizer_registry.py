from torch.optim import AdamW, Adam, SGD


OPTIMIZER_REGISTRY: list[type] = [
    AdamW,
    Adam,
    SGD,
]
