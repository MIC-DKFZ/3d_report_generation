from torch.optim import AdamW, Adam, SGD

from vlm.training.optimizers.muon import MuonWithAdamW


OPTIMIZER_REGISTRY: list[type] = [
    AdamW,
    Adam,
    SGD,
    MuonWithAdamW,
]
