import torch
import torch.nn as nn
from einops import rearrange

from vlm.adapters.layers.dense import MixerBlockNoSkips, MixerBlock
from vlm.adapters.query_adapters.base import QueryAdapter


class MLPMixerTextEmbedDim(QueryAdapter):

    def __init__(self, depth: int, **kwargs):
        super().__init__(**kwargs)

        lis: list[nn.Module] = []
        tokens_and_features: list[tuple[int, int]] = [(self.num_vision_tokens, self.vision_embed_dim)]

        for _ in range(depth-1):
            tokens_and_features.append((self.num_query_tokens, self.text_embed_dim))

        tokens_and_features.append((self.num_query_tokens, self.text_embed_dim))

        for i in range(len(tokens_and_features) - 1):
            in_vision_tokens, in_features = tokens_and_features[i]
            out_vision_tokens, out_features = tokens_and_features[i + 1]
            if i == 0:
                mixer_block_class = MixerBlockNoSkips
            elif i == len(tokens_and_features) - 2:
                mixer_block_class = MixerBlockNoSkips
            else:
                mixer_block_class = MixerBlock
            lis.append(
                mixer_block_class(in_features, out_features, in_vision_tokens, out_vision_tokens,
                                  1, 2, self.norm_layer_class, self.bias)
            )
        self.mixer_blocks = nn.Sequential(*lis)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        # batch_size, embed_dim, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> b (t h w) c").contiguous()

        x = self.mixer_blocks(x)

        return x


class MLPMixerVisionEmbedDim(QueryAdapter):

    def __init__(self, depth: int, **kwargs):
        super().__init__(**kwargs)

        lis: list[nn.Module] = []
        tokens_and_features: list[tuple[int, int]] = [(self.num_vision_tokens, self.vision_embed_dim)]

        for _ in range(depth-1):
            tokens_and_features.append((self.num_query_tokens, self.vision_embed_dim))

        tokens_and_features.append((self.num_query_tokens, self.text_embed_dim))

        for i in range(len(tokens_and_features) - 1):
            in_vision_tokens, in_features = tokens_and_features[i]
            out_vision_tokens, out_features = tokens_and_features[i + 1]
            if i == len(tokens_and_features) - 2:
                mixer_block_class = MixerBlockNoSkips
                channel_mix_factor = 2
            elif i == 0:
                mixer_block_class = MixerBlockNoSkips
                channel_mix_factor = 4
            else:
                mixer_block_class = MixerBlock
                channel_mix_factor = 4
            lis.append(
                mixer_block_class(in_features, out_features, in_vision_tokens, out_vision_tokens,
                                  1, channel_mix_factor, self.norm_layer_class, self.bias)
            )
        self.mixer_blocks = nn.Sequential(*lis)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        # batch_size, embed_dim, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> b (t h w) c").contiguous()

        x = self.mixer_blocks(x)

        return x


class MLPMixerNoSkips(QueryAdapter):

    def __init__(self, depth: int, **kwargs):
        super().__init__(**kwargs)

        lis: list[nn.Module] = []
        tokens_and_features: list[tuple[int, int]] = [(self.num_vision_tokens, self.vision_embed_dim)]

        for _ in range(depth-1):
            tokens_and_features.append((self.num_query_tokens, self.vision_embed_dim))

        tokens_and_features.append((self.num_query_tokens, self.text_embed_dim))

        for i in range(len(tokens_and_features) - 1):
            in_vision_tokens, in_features = tokens_and_features[i]
            out_vision_tokens, out_features = tokens_and_features[i + 1]
            channel_mix_factor = 1 if i == 0 else 4
            lis.append(
                MixerBlockNoSkips(in_features, out_features, in_vision_tokens, out_vision_tokens,
                                  1, channel_mix_factor, self.norm_layer_class, self.bias)
            )
        self.mixer_blocks = nn.Sequential(*lis)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        # batch_size, embed_dim, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> b (t h w) c").contiguous()

        x = self.mixer_blocks(x)

        return x
