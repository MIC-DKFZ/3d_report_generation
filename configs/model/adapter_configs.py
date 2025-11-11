from abc import ABC, abstractmethod
from typing import Literal
from pydantic import Field

from adapters.grid_adapters.pool import AveragePool, MaxPool
from adapters.grid_adapters.qwen_adapter import QwenAdapter
from adapters.query_adapters.attention_pool import AttentionPool
from configs.base import DiscriminatedConfigBase


class DiscriminatedAdapterConfigBase(DiscriminatedConfigBase, ABC):
    @abstractmethod
    def to_adapter(self, *args, **kwargs):
        """Return the adapter object"""
        ...


# ====================
# Grid Adapters
# ====================

class GridAdapterConfigBase(DiscriminatedAdapterConfigBase, ABC):
    reduce_block: tuple[int, int, int] = Field((2, 2, 2), description="Token reduce factor for each axis.")

# --------------------

class QwenAdapterConfig(GridAdapterConfigBase):
    type: Literal["qwen"]

    def to_adapter(self, vision_embed_dim: int, text_embed_dim: int) -> QwenAdapter:
        return QwenAdapter(vision_embed_dim=vision_embed_dim,
                           text_embed_dim=text_embed_dim,
                           reduce_block=self.reduce_block)


class AveragePoolAdapterConfig(GridAdapterConfigBase):
    type: Literal["average"]

    def to_adapter(self, vision_embed_dim: int, text_embed_dim: int) -> AveragePool:
        return AveragePool(vision_embed_dim, text_embed_dim, self.reduce_block)


class MaxPoolAdapterConfig(GridAdapterConfigBase):
    type: Literal["max"]

    def to_adapter(self, vision_embed_dim: int, text_embed_dim: int) -> MaxPool:
        return MaxPool(vision_embed_dim, text_embed_dim, self.reduce_block)


# ====================
# Query Adapters
# ====================

class QueryAdapterConfigBase(DiscriminatedAdapterConfigBase, ABC):
    num_query_tokens: int = Field(64, description="Number of Query Tokens")

# --------------------

class AttentionPoolConfig(QueryAdapterConfigBase):
    type: Literal["attention"]

    def to_adapter(self, vision_embed_dim: int, text_embed_dim: int) -> AttentionPool:
        return AttentionPool(vision_embed_dim, text_embed_dim, self.num_query_tokens)


class ResamplerConfig(QueryAdapterConfigBase):
    type: Literal["resampler"]
    n_layers: int = Field(4, description="")

    def to_adapter(self, vision_embed_dim: int, text_embed_dim: int) -> None:
        ...

