from adapters.grid_adapters.base import GridAdapter


class AveragePool(GridAdapter):
    """

    """
    def __init__(
        self,
        vision_embed_dim: int,
        text_embed_dim: int,
        reduce_block: tuple[int, int, int]
    ):
        super().__init__(
            vision_embed_dim=vision_embed_dim,
            text_embed_dim=text_embed_dim,
            reduce_block=reduce_block
        )


    def forward(self, x):
        ...


class MaxPool(GridAdapter):
    """

    """
    def __init__(
        self,
        vision_embed_dim: int,
        text_embed_dim: int,
        reduce_block: tuple[int, int, int]
    ):
        super().__init__(
            vision_embed_dim=vision_embed_dim,
            text_embed_dim=text_embed_dim,
            reduce_block=reduce_block
        )


    def forward(self, x):
        ...