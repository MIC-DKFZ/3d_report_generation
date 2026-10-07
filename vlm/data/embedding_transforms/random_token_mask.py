import random

import numpy as np

from vlm.utils.data.embedding_transforms import (
    clone_embedding,
    should_apply,
    validate_embedding,
    validate_probability,
)


class RandomTokenMask:
    def __init__(
        self,
        p_apply: float = 0.5,
        min_mask_ratio: float = 0.15,
        max_mask_ratio: float = 0.30,
    ):
        validate_probability("p_apply", p_apply)
        validate_probability("min_mask_ratio", min_mask_ratio)
        validate_probability("max_mask_ratio", max_mask_ratio)
        if min_mask_ratio > max_mask_ratio:
            raise ValueError("min_mask_ratio must be <= max_mask_ratio")

        self.p_apply = p_apply
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.grid_size: tuple[int, int, int] | None = None

    def _get_grid_size(self, embedding) -> tuple[int, int, int]:
        if embedding.ndim < 3:
            raise ValueError(
                "RandomTokenMask expects spatial dimensions on the last three axes. "
                f"Got embedding shape {tuple(embedding.shape)}."
            )

        grid_size = tuple(embedding.shape[-3:])
        if grid_size[0] <= 0 or grid_size[1] <= 0 or grid_size[2] <= 0:
            raise ValueError(f"Invalid embedding spatial shape {grid_size}.")

        if self.grid_size is None:
            self.grid_size = grid_size
        elif grid_size != self.grid_size:
            raise ValueError(
                f"RandomTokenMask cached spatial shape {self.grid_size}, "
                f"but got {grid_size}."
            )

        return self.grid_size

    def __call__(self, embedding):
        embedding = validate_embedding(embedding)
        grid_d, grid_h, grid_w = self._get_grid_size(embedding)

        if not should_apply(self.p_apply):
            return embedding

        output = clone_embedding(embedding)
        mask_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        grid_volume = grid_d * grid_h * grid_w
        n_mask_tokens = max(1, min(grid_volume, round(mask_ratio * grid_volume)))

        flat_indices = np.random.choice(grid_volume, size=n_mask_tokens, replace=False)
        depth_indices, row_indices, col_indices = np.unravel_index(
            flat_indices,
            (grid_d, grid_h, grid_w),
        )

        output[..., depth_indices, row_indices, col_indices] = 0
        return output
