import torch
from einops import rearrange

from vlm.adapters.query_adapters.base import QueryAdapter
from vlm.adapters.layers.dense import MLP


class OneToOneAdapterSeq(QueryAdapter):
    """
    Project sequence vision tokens one-to-one without a 3D spatial grid, e.g. Jolia.
    Cannot use the default OneToOneAdapter, as it uses 3D positional embeddings if LLM uses them.
    """

    def __init__(self, depth: int, **kwargs):
        super().__init__(**kwargs)
        input_dim = self.vision_embed_dim
        self.norm = self.norm_layer_class(input_dim)

        self.dense_projector = MLP(in_features=self.vision_embed_dim, out_features=self.text_embed_dim, depth=depth)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        # batch_size, embed_dim, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> b (t h w) c").contiguous()

        # Process through adapter
        x = self.norm(x)
        x = self.dense_projector(x)

        return x
