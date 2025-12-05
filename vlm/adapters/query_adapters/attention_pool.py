import torch.nn as nn
from vlm.adapters.query_adapters.base import QueryAdapter


class AttentionPool(QueryAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, x):
        ...