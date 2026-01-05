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
                 hidden_dim_factor: int, norm_layer_class: type, bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_vision_tokens = in_vision_tokens
        self.out_vision_tokens = out_vision_tokens

        self.pre_token_mix_norm = norm_layer_class(in_features)
        self.token_mix_mlp = MLP(features=[in_vision_tokens, hidden_dim_factor * in_vision_tokens, out_vision_tokens], bias=bias)

        self.pre_channel_mix_norm = norm_layer_class(in_features)
        self.channel_mix_mlp = MLP(features=[in_features, hidden_dim_factor*in_features, out_features], bias=bias)

    def forward(self, x):
        x = self.pre_token_mix_norm(x)

        x = rearrange(x, "b n d -> b d n")

        x = self.token_mix_mlp(x)

        x = rearrange(x, "b d n -> b n d")

        x = self.pre_channel_mix_norm(x)
        x = self.channel_mix_mlp(x)

        return x