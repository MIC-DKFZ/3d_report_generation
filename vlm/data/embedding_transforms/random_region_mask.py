import random
from vlm.data.embedding_transforms._helpers import (
    clone_embedding,
    should_apply,
    validate_embedding,
    validate_probability,
)


class RandomRegionMask:
    def __init__(
        self,
        p_apply: float = 0.5,
        min_mask_ratio: float = 0.10,
        max_mask_ratio: float = 0.25,
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
                "RandomRegionMask expects spatial dimensions on the last three axes. "
                f"Got embedding shape {tuple(embedding.shape)}."
            )

        grid_size = tuple(embedding.shape[-3:])
        if grid_size[0] <= 0 or grid_size[1] <= 0 or grid_size[2] <= 0:
            raise ValueError(f"Invalid embedding spatial shape {grid_size}.")

        if self.grid_size is None:
            self.grid_size = grid_size
        elif grid_size != self.grid_size:
            raise ValueError(
                f"RandomRegionMask cached spatial shape {self.grid_size}, "
                f"but got {grid_size}."
            )

        return self.grid_size

    def __call__(self, embedding):
        embedding = validate_embedding(embedding)
        grid_d, grid_h, grid_w = self._get_grid_size(embedding)

        if not should_apply(self.p_apply):
            return embedding

        output = clone_embedding(embedding)
        area_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        grid_volume = grid_d * grid_h * grid_w
        target_volume = max(1, min(grid_volume, round(area_ratio * grid_volume)))

        base_size = target_volume ** (1.0 / 3.0)
        depth_ratio = random.uniform(0.5, 2.0)
        height_ratio = random.uniform(0.5, 2.0)
        mask_d = max(1, min(grid_d, round(base_size * depth_ratio)))
        mask_h = max(1, min(grid_h, round(base_size * height_ratio)))
        mask_w = max(1, min(grid_w, round(target_volume / (mask_d * mask_h))))
        mask_d = max(1, min(grid_d, mask_d))
        mask_h = max(1, min(grid_h, mask_h))
        mask_w = max(1, min(grid_w, mask_w))

        depth_start = random.randint(0, grid_d - mask_d)
        row_start = random.randint(0, grid_h - mask_h)
        col_start = random.randint(0, grid_w - mask_w)
        depth_end = depth_start + mask_d
        row_end = row_start + mask_h
        col_end = col_start + mask_w

        output[..., depth_start:depth_end, row_start:row_end, col_start:col_end] = 0
        return output
