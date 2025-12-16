import torch
from vlm.adapters.grid_adapters.base import GridAdapter
from vlm.adapters.layers.dense import MLP


class ConcatAdapter(GridAdapter):
    def __init__(self, depth: int, **kwargs):
        super().__init__(**kwargs)
        input_dim = self.vision_embed_dim * self.reduce_block[0] ** 3
        self.norm = self.norm_layer_class(input_dim)

        def round_to_multiple(x, multiple: int = 64):
            return ((x + multiple - 1) // multiple) * multiple

        first_hidden_dim = max(
            self.text_embed_dim,
            min(round_to_multiple(self.vision_embed_dim * self.reduce_block[0] ** 2), 4 * self.text_embed_dim)
        )

        remaining_hidden_dims = []
        if depth > 1:
            step = (first_hidden_dim - self.text_embed_dim) / depth
            for i in range(1, depth):
                next_dim = max(self.text_embed_dim, int(first_hidden_dim - (step * i)))
                if next_dim > self.text_embed_dim:
                    next_dim = round_to_multiple(next_dim)
                remaining_hidden_dims.append(next_dim)
        all_dims = [input_dim, first_hidden_dim, *remaining_hidden_dims, self.text_embed_dim]

        self.dense_projector = MLP(features=all_dims, bias=self.bias)

    def forward(self, x: torch.Tensor, adapter_args: dict):
        batch_size, embed_dim, T, H, W = x.shape

        r0, r1, r2 = self.reduce_block
        # Use unfold to create reduce_block
        x = x.unfold(2, r0, r0)
        x = x.unfold(3, r1, r1)
        x = x.unfold(4, r2, r2)

        # Reshape to combine all reduce_block
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7)  # Move embed_dim after spatial dims
        # (B, T', H', W', C, rT, rH, rW) -> (B, T'*H'*W', C*rT*rH*rW)
        x = x.reshape(batch_size, -1, embed_dim * r0 * r1 * r2)

        # Process through adapter
        # x = x.reshape(-1, embed_dim * self.reduce_block[0] * self.reduce_block[1] * self.reduce_block[2])
        x = self.norm(x)
        x = self.dense_projector(x)

        return x
