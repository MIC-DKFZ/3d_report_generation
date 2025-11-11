from adapters.query_adapters.base import QueryAdapter


class AttentionPool(QueryAdapter):
    def __init__(
        self,
        vision_embed_dim: int,
        text_embed_dim: int,
        num_query_tokens: int
    ):
        super().__init__(
            vision_embed_dim=vision_embed_dim,
            text_embed_dim=text_embed_dim,
            num_query_tokens=num_query_tokens
        )


    def forward(self, x):
        ...