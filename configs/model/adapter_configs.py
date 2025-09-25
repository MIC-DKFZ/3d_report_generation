from abc import ABC, abstractmethod
from typing import Literal

from pydantic import Field
from configs.base import DiscriminatedConfigBase
from adapters.adapters import VisionAdapterReduceBlocks


class DiscriminatedAdapterConfigBase(DiscriminatedConfigBase, ABC):
    @abstractmethod
    def to_adapter(self, *args, **kwargs):
        """Return the adapter object"""
        ...


class QwenAdapterConfig(DiscriminatedAdapterConfigBase):
    type: Literal["qwen"]
    reduce_block: tuple[int, int, int] = Field((2, 2, 2), description="Token reduce factor for each axis.")

    def to_adapter(self, vision_embed_dim: int, text_embed_dim: int) -> VisionAdapterReduceBlocks:
        return VisionAdapterReduceBlocks(embedding_dim=vision_embed_dim,
                                         embed_dim=text_embed_dim,
                                         reduce_block=self.reduce_block)