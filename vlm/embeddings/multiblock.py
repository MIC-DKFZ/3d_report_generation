"""
Multi-block encoder inputs: a window of (gz, gy, gx) encoder inputs, encoded block by block and
merged into one embedding grid (e.g. the Primus-4B embeddings of the RMR paper).

The window is the single-block encoder input ('preprocess_encoder_input') with the window as input
shape, tiled with nnU-Net sliding-window steps: with 'tile_step_size' 1 into gz·gy·gx contiguous
blocks, with smaller steps into overlapping blocks, whose embeddings are merged with Gaussian
weights.
"""

import SimpleITK as sitk
import torch

from vlm.data.preprocessing.encoder_input import preprocess_encoder_input
from vlm.utils.data.embeddings import compute_gaussian, compute_steps_for_sliding_window

Shape = tuple[int, int, int]


def window_shape(block: Shape, grid: Shape) -> Shape:
    return tuple(b * g for b, g in zip(block, grid))


def block_offsets(block: Shape, grid: Shape, tile_step_size: float) -> list[Shape]:
    """Window-local (z, y, x) offsets of the blocks, row-major."""
    if not 0 < tile_step_size <= 1:
        raise ValueError("tile_step_size must be in (0, 1]")
    steps = compute_steps_for_sliding_window(window_shape(block, grid), block, tile_step_size)
    return [(z, y, x) for z in steps[0] for y in steps[1] for x in steps[2]]


def tile_window(window: torch.Tensor, block: Shape, offsets: list[Shape]) -> torch.Tensor:
    """(c, *window) → (n_blocks, c, *block)."""
    bz, by, bx = block
    return torch.stack([window[:, z : z + bz, y : y + by, x : x + bx] for z, y, x in offsets])


def preprocess_multiblock_input(
    image: sitk.Image,
    mask: sitk.Image | None,
    *,
    input_shape: Shape,
    grid: Shape,
    tile_step_size: float = 1.0,
    **preprocessing_kwargs,
) -> tuple[torch.Tensor, list[Shape]]:
    """
    The blocks (n_blocks, 1, *input_shape) of a volume and their window-local offsets.
    'preprocessing_kwargs' are those of 'preprocess_encoder_input' besides the input shape.
    """
    window = window_shape(input_shape, grid)
    offsets = block_offsets(input_shape, grid, tile_step_size)
    data = preprocess_encoder_input(image, mask, input_shape=window, **preprocessing_kwargs)[0]
    return tile_window(data, input_shape, offsets), offsets


def merge_block_embeddings(
    blocks: torch.Tensor,
    offsets: list[Shape],
    input_shape: Shape,
    grid: Shape,
    use_gaussian: bool = True,
) -> torch.Tensor:
    """
    Merge the (n_blocks, C, d, h, w) block embeddings into one (C, gz·d, gy·h, gx·w) grid. The
    window-local voxel offsets are scaled onto the embedding grid; overlapping blocks are
    averaged with nnU-Net's Gaussian weights (uniform weights without 'use_gaussian'). Without
    overlap, every embedding voxel comes from one block.
    """
    n, c, *size = blocks.shape
    if n != len(offsets):
        raise ValueError(f"Expected {len(offsets)} blocks, got {n}")
    canvas_shape = tuple(s * g for s, g in zip(size, grid))
    canvas = blocks.new_zeros((c, *canvas_shape))
    weight = blocks.new_zeros((1, *canvas_shape))
    if use_gaussian:
        g = compute_gaussian(
            tuple(size), sigma_scale=1.0 / 8, dtype=blocks.dtype, device=blocks.device
        )
    else:
        g = blocks.new_ones(tuple(size))
    for block, offset in zip(blocks, offsets):
        start = [round(o * s / p) for o, s, p in zip(offset, size, input_shape)]
        region = tuple(slice(st, st + s) for st, s in zip(start, size))
        canvas[(slice(None), *region)] += block * g
        weight[(slice(None), *region)] += g
    return canvas / weight
