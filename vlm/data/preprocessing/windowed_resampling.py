"""
Resampling of the encoder input's region only, with the result of resampling the whole volume.

'resample_to_spacing' (vlm/data/preprocessing/encoder_input.py) resizes with skimage
'resize(order=3, mode="edge", anti_aliasing=False, clip=True)' to new_shape =
round(shape * spacing / target_spacing): output voxel o of an axis samples the input at
s * (o + 0.5) - 0.5, s = shape / new_shape, and the result is clipped to the input's range.
Segmentations are resized the same way, label by label, and thresholded at 0.5.

The cubic spline coefficients of a voxel depend on its neighbourhood with a weight that decays by
0.268 per voxel, so a region resampled from its input region plus MARGIN voxels matches the whole
volume's resampling to about 0.268 ** MARGIN of the intensity range.
"""

import math

import numpy as np
from scipy import ndimage as ndi

from vlm.data.preprocessing.crop_to_mask import (
    enforce_min_size,
    expand_bounding_box,
    get_mask_bounding_box,
)
from vlm.data.preprocessing.default_resampling import compute_new_shape

MARGIN = 16

Shape = tuple[int, int, int]
Window = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def resample_window(
    data: np.ndarray,
    new_shape: Shape,
    window: Window,
    value_range: tuple[float, float],
    order: int = 3,
) -> np.ndarray:
    """
    The output voxels [a, b) per axis of resizing the (z, y, x) array 'data' to 'new_shape', as
    float64, clipped to 'value_range' (the whole input's minimum and maximum).
    """
    scales = [n / o for n, o in zip(data.shape, new_shape)]
    region, offset = [], []
    for s, (a, b), n in zip(scales, window, data.shape):
        lo = max(0, math.floor(s * (a + 0.5) - 0.5) - MARGIN)
        hi = min(n, math.ceil(s * (b - 0.5) - 0.5) + MARGIN + 1)
        region.append(slice(lo, hi))
        offset.append(s * (a + 0.5) - 0.5 - lo)
    out = ndi.affine_transform(
        np.asarray(data[tuple(region)], dtype=np.float64),
        np.diag(scales),
        offset=offset,
        output_shape=tuple(b - a for a, b in window),
        order=order,
        mode="nearest",
    )
    return np.clip(out, *value_range, out=out)


def output_window(lo: int, hi: int, scale: float, n_out: int, pad: int = 2) -> tuple[int, int]:
    """The output voxels of an axis whose sample points fall into input voxels [lo-pad, hi+pad)."""
    a = max(0, math.floor((lo - pad + 0.5) / scale - 0.5))
    b = min(n_out, math.ceil((hi + pad + 0.5) / scale - 0.5) + 1)
    return a, b


def resampled_mask_bbox(mask: np.ndarray, new_shape: Shape) -> list[int] | None:
    """
    The bounding box (z0, z1, y0, y1, x0, x1) of the binary (z, y, x) mask resized to 'new_shape'
    as a segmentation; None if the resized mask is empty. Only a window around the mask's bounding
    box is resized.
    """
    fg = mask > 0
    if not fg.any():
        return None
    box = get_mask_bounding_box(fg)
    scales = [n / o for n, o in zip(mask.shape, new_shape)]
    window = [output_window(box[2 * i], box[2 * i + 1], scales[i], new_shape[i]) for i in range(3)]
    resized = resample_window(fg, new_shape, window, (0.0, 1.0)) >= 0.5
    if not resized.any():
        return None
    local_box = get_mask_bounding_box(resized)
    out = []
    for i, (a, b) in enumerate(window):
        lo, hi = int(local_box[2 * i]), int(local_box[2 * i + 1])
        # The window holds all of the resized mask unless it touches an inner window edge.
        if (lo == 0 and a > 0) or (hi == b - a and b < new_shape[i]):
            raise RuntimeError("The resampled mask reaches beyond its window.")
        out += [lo + a, hi + a]
    return out


def resampled_roi(
    data: np.ndarray,
    mask: np.ndarray | None,
    spacing: Shape,
    mask_spacing: Shape,
    target_spacing: Shape,
    min_size_zyx: Shape | None,
    crop_to: Shape | None,
) -> np.ndarray | None:
    """
    The ROI of the (1, z, y, x) data resampled to 'target_spacing' and cropped by
    'crop_to_mask_roi' with the mask resampled likewise (the whole volume without a mask),
    resampling only the voxels of the ROI. With 'crop_to', only the ROI's center that
    'center_crop_or_pad_3d' keeps for that shape. None where the shortcut does not apply: the
    data needs no resampling, the mask is on another grid, or it is empty.
    """
    if np.allclose(spacing, target_spacing, atol=1e-2):
        return None
    new_shape = tuple(int(n) for n in compute_new_shape(data.shape[1:], spacing, target_spacing))
    if mask is None:
        box = [0, new_shape[0], 0, new_shape[1], 0, new_shape[2]]
    else:
        if mask.shape != data.shape or np.allclose(mask_spacing, target_spacing, atol=1e-2):
            return None
        if tuple(compute_new_shape(mask.shape[1:], mask_spacing, target_spacing)) != new_shape:
            return None
        box = resampled_mask_bbox(mask[0], new_shape)
        if box is None:
            return None
        box = expand_bounding_box(box, new_shape, 0.04)
        if min_size_zyx:
            box = enforce_min_size(box, min_size_zyx, new_shape)
    window = []
    for i in range(3):
        a, b = box[2 * i], box[2 * i + 1]
        if crop_to is not None and b - a > crop_to[i]:
            a += (b - a - crop_to[i]) // 2
            b = a + crop_to[i]
        window.append((a, b))
    value_range = (float(data.min()), float(data.max()))
    return resample_window(data[0], new_shape, window, value_range).astype(data.dtype)[None]
