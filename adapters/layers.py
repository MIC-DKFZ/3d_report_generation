from torch import nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, depth: int):
        super().__init__()
        lis: list[nn.Module] = [nn.Linear(in_features=in_features, out_features=out_features)]

        for i in range(depth):
            lis.append(GELULinear(in_features=out_features, out_features=out_features))

        self.mlp = nn.Sequential(*lis)

    def forward(self, x):
        return self.mlp(x)

class GELULinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.act = nn.GELU()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        return self.linear(self.act(x))


class MixerBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, depth: int, num_vision_tokens: int):
        super().__init__()

    def forward(self, x):
        ...
