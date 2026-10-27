import numpy as np

from vlm.data.embedding_transforms._helpers import (
    should_apply,
    validate_embedding,
    validate_probability,
)


class GaussianNoise:
    def __init__(
        self,
        p_apply: float = 0.5,
        sigma: float = 0.03,
        relative_sigma: bool = True,
    ):
        validate_probability("p_apply", p_apply)
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")

        self.p_apply = p_apply
        self.sigma = sigma
        self.relative_sigma = relative_sigma

    def __call__(self, embedding):
        if not should_apply(self.p_apply):
            return embedding

        embedding = validate_embedding(embedding)
        sigma = self.sigma
        if self.relative_sigma:
            sigma = sigma * float(embedding.astype(np.float32, copy=False).std())

        noise = np.random.normal(0.0, sigma, size=embedding.shape).astype(
            embedding.dtype,
            copy=False,
        )
        return embedding + noise
