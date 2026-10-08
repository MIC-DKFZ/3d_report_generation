import numpy as np

from vlm.data.embedding_transforms._helpers import (
    clone_embedding,
    should_apply,
    validate_embedding,
    validate_probability,
)


class MixUp:
    def __init__(
        self,
        p_apply: float = 0.5,
        min_current_weight: float = 0.75,
        max_current_weight: float = 0.9,
    ):
        validate_probability("p_apply", p_apply)
        validate_probability("min_current_weight", min_current_weight)
        validate_probability("max_current_weight", max_current_weight)

        if min_current_weight > max_current_weight:
            raise ValueError("min_current_weight must be <= max_current_weight")

        self.p_apply = p_apply
        self.min_current_weight = min_current_weight
        self.max_current_weight = max_current_weight
        self._last_embedding = None

    def __call__(self, embedding):
        embedding = validate_embedding(embedding)
        previous_embedding = self._last_embedding
        self._last_embedding = clone_embedding(embedding)

        if previous_embedding is None or not should_apply(self.p_apply):
            return embedding

        if tuple(previous_embedding.shape) != tuple(embedding.shape):
            return embedding

        lam = float(
            np.random.uniform(
                self.min_current_weight,
                self.max_current_weight,
            )
        )

        previous_embedding = previous_embedding.astype(embedding.dtype, copy=False)
        mixed = lam * embedding + (1.0 - lam) * previous_embedding

        return mixed.astype(embedding.dtype, copy=False)
