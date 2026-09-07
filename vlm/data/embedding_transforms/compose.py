from __future__ import annotations

from typing import Protocol


class EmbeddingTransform(Protocol):
    def __call__(self, embedding):
        ...


class EmbeddingCompose:
    def __init__(self, transforms: list[EmbeddingTransform]):
        self.transforms = transforms

    def __call__(self, embedding):
        for transform in self.transforms:
            embedding = transform(embedding)
        return embedding
