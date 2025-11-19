from torch import nn as nn
from einops import rearrange


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
    def __init__(self, in_features: int, out_features: int, in_vision_tokens: int, out_vision_tokens: int, depth: int,):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.depth = depth
        self.in_vision_tokens = in_vision_tokens
        self.out_vision_tokens = out_vision_tokens

        self.token_layernorm = nn.LayerNorm(in_features)
        self.channel_mlp = MLP(in_vision_tokens, out_vision_tokens, depth=1)

        self.channel_layernorm = nn.LayerNorm(out_vision_tokens)
        self.token_mlp = MLP(in_features, out_features, depth=1)

    def forward(self, x):
        x = self.token_layernorm(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.channel_mlp(x)

        x = self.channel_layernorm(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.token_mlp(x)

        return x






