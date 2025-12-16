from torch import nn as nn
from einops import rearrange


class MLP(nn.Sequential):
    def __init__(self, in_features: int | None = None, out_features: int | None = None, depth: int | None = None,
                 features: list[int] | None = None, bias: bool = True):

        lis: list[nn.Module] = []

        # Case 1: Fixed hidden size
        if in_features and out_features and depth >= 0 and features is None:
            lis.append(nn.Linear(in_features, out_features, bias=bias))
            for _ in range(depth):
                lis.append(nn.SiLU())  # Add activation separately
                lis.append(nn.Linear(out_features, out_features, bias=bias))

        # Case 2: Custom feature sizes per layer
        elif features and in_features is None and out_features is None and depth is None:
            for i in range(len(features)-1):
                if i > 0:
                    lis.append(nn.SiLU())
                lis.append(nn.Linear(features[i], features[i + 1], bias=bias))
        else:
            raise ValueError("Invalid arguments provided.")

        super().__init__(*lis)


class MixerBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, in_vision_tokens: int, out_vision_tokens: int,
                 norm_fn: str = "RMSNorm", bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_vision_tokens = in_vision_tokens
        self.out_vision_tokens = out_vision_tokens

        self.token_norm = getattr(nn, norm_fn)(in_features)
        self.channel_mlp = MLP(in_vision_tokens, out_vision_tokens, depth=1, bias=bias)

        self.channel_norm = getattr(nn, norm_fn)(out_vision_tokens)
        self.token_mlp = MLP(in_features, out_features, depth=1, bias=bias)

    def forward(self, x):
        x = self.token_norm(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.channel_mlp(x)

        x = self.channel_norm(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.token_mlp(x)

        return x


class MLPMixer(nn.Module):
    def __init__(self, in_features: int, out_features: int, in_vision_tokens: int, out_vision_tokens: int,
                 norm_fn: str = "RMSNorm", bias: bool = True):
        pass
