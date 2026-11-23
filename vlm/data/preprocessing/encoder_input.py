"""
Preprocessing of CT volumes into vision encoder inputs, shared by the embedding generation
(vlm/scripts/embeddings/generate_embeddings.py) and the inference on NIfTI images.

Single-block pipeline per volume: normalize → resample to the encoder spacing, if it has one →
crop to the mask ROI (bounding box expanded by 4%) → normalize again (z_score, min_max) → fit to the encoder input
shape by resizing ('resized') or center cropping/padding ('center_crop'). Only the voxels of the
ROI (with 'center_crop', of the part that is kept) are resampled, with the result of resampling
the whole volume (vlm/data/preprocessing/windowed_resampling.py).

'resized_direct' skips the resampling to the target spacing: the ROI is cropped at the volume's
own spacing and resized to the encoder input shape in one step, with a Gaussian anti-aliasing
filter along the axes it shrinks. The target spacing then only sets the smallest ROI extent of
'enforce_min_bbox' (the input shape at that spacing, in mm), as in 'resized'.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from vlm.data.preprocessing.crop_to_mask import (
    enforce_min_size,
    expand_bounding_box,
    get_mask_bounding_box,
)
from vlm.data.preprocessing.default_resampling import (
    compute_new_shape,
    resample_data_or_seg,
)
from vlm.data.preprocessing.windowed_resampling import resampled_roi

PROCESSING_MODES = ("resized", "center_crop", "resized_direct")
NORMALIZATIONS = (
    "z_score",
    "min_max",
    "vista3d_scale",
    "merlin_scale",
    "fvlm_scale",
    "radar_scale",
    "none",
)
# HU windows mapped to [0, 1] (clipped). They are applied once, before the ROI crop; z_score and
# min_max are computed again on the ROI.
HU_WINDOWS = {
    "vista3d_scale": (-963.8247715525971, 1053.678477684517),
    "merlin_scale": (-1000.0, 1000.0),
    "fvlm_scale": (-1150.0, 350.0),
    "radar_scale": (-300.0, 400.0),
}
OUTLIER_METHODS = ("none", "connected_components")


def remove_mask_outliers(
    mask: np.ndarray,
    method: str = "connected_components",
    min_voxels: int = 400
) -> np.ndarray:
    """
    Remove outlier voxels from a binary mask using connected component analysis.

    Args:
        mask: Binary mask array (Z, Y, X) or (1, Z, Y, X)
        method: 'connected_components' or 'none'
        min_voxels: Minimum voxel count of connected components that are kept

    Returns:
        Cleaned binary mask with same shape as input
    """
    if method == "none":
        return mask

    # Handle channel dimension
    squeeze = False
    if mask.ndim == 4:
        squeeze = True
        mask = mask[0]

    mask = mask.astype(np.uint8)

    labeled, num_features = ndi.label(mask)
    if num_features == 0:
        cleaned = mask
    else:
        component_sizes = ndi.sum(mask, labeled, range(1, num_features + 1))
        keep_labels = np.where(component_sizes >= min_voxels)[0] + 1
        cleaned = np.isin(labeled, keep_labels).astype(np.uint8)


    if squeeze:
        cleaned = cleaned[np.newaxis, ...]

    return cleaned


def binarize_mask(
    mask: np.ndarray,
    mask_label_ids: Optional[List[int]],
    outlier_method: str,
    outlier_min_voxels: int,
) -> np.ndarray:
    """Binary mask of the given labels (any non-zero label without labels), outliers removed."""
    if mask_label_ids is None or len(mask_label_ids) == 0:
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = np.isin(mask, mask_label_ids).astype(np.uint8)

    if outlier_method != "none":
        mask = remove_mask_outliers(mask, method=outlier_method, min_voxels=outlier_min_voxels)
    return mask


def resample_to_spacing(
    data: np.ndarray,
    current_spacing: Tuple[float, ...],
    target_spacing: Tuple[float, ...],
    is_seg: bool = False,
) -> np.ndarray:
    """
    Resample (c, z, y, x) data to the target spacing with nnU-Net's 'resample_data_or_seg'
    (order 3, never separately along z), unless it already has it (within 0.01 mm).
    """
    if np.allclose(current_spacing, target_spacing, atol=1e-2):
        return data
    new_shape = compute_new_shape(data.shape[1:], current_spacing, target_spacing)
    return resample_data_or_seg(data, new_shape, is_seg=is_seg)


def apply_normalization(image: np.ndarray, scheme: str = "z_score") -> np.ndarray:
    """Apply normalization to image data. Input shape: (C, Z, Y, X)."""
    image = image.astype(np.float32)
    if scheme == "z_score":
        mean = image.mean()
        std = image.std()
        image = (image - mean) / max(std, 1e-8)
        return image

    elif scheme == "min_max":
        min_val, max_val = image.min(), image.max()
        if max_val - min_val > 1e-8:
            image = (image - min_val) / (max_val - min_val)
        return image

    elif scheme in HU_WINDOWS:
        a_min, a_max = HU_WINDOWS[scheme]
        image = (image - a_min) / (a_max - a_min)
        image = np.clip(image, 0.0, 1.0)
        return image

    elif scheme == "none":
        return image

    else:
        raise ValueError(f"Unknown normalization scheme: {scheme}")


def center_crop_or_pad_3d(
    x: torch.Tensor, target_size: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Center-crop if image > target, center-pad with zeros if image < target.
    Input shape: (B, C, Z, Y, X). Returns: (B, C, tZ, tY, tX).
    """
    assert x.ndim == 5, f"Expected 5D tensor [B, C, Z, Y, X], got {x.shape}"

    B, C, Z, Y, X = x.shape
    tZ, tY, tX = target_size

    out = x.new_zeros((B, C, tZ, tY, tX))

    def compute_indices(orig, target):
        copy_len = min(orig, target)
        src_start = max((orig - target) // 2, 0)
        src_end = src_start + copy_len
        dst_start = max((target - orig) // 2, 0)
        dst_end = dst_start + copy_len
        return src_start, src_end, dst_start, dst_end

    z_s0, z_s1, z_d0, z_d1 = compute_indices(Z, tZ)
    y_s0, y_s1, y_d0, y_d1 = compute_indices(Y, tY)
    x_s0, x_s1, x_d0, x_d1 = compute_indices(X, tX)

    out[:, :, z_d0:z_d1, y_d0:y_d1, x_d0:x_d1] = x[
        :, :, z_s0:z_s1, y_s0:y_s1, x_s0:x_s1
    ]
    return out


def spacing_zyx(image: sitk.Image) -> Tuple[float, float, float]:
    """The spacing of 'image' in the (z, y, x) order of its array (SimpleITK reports (x, y, z))."""
    return image.GetSpacing()[::-1]


def crop_to_mask_roi(
    data: np.ndarray, mask: np.ndarray, min_size_zyx: Optional[Tuple[int, int, int]]
) -> np.ndarray:
    """
    Crop (c, z, y, x) data to the mask bounding box, expanded by 4% and, with 'min_size_zyx',
    enlarged to at least that size where the image allows it.
    """
    box = get_mask_bounding_box(mask[0])
    box = expand_bounding_box(box, data.shape[1:], 0.04)
    if min_size_zyx:
        box = enforce_min_size(box, min_size_zyx, data.shape[1:])
    z0, z1, y0, y1, x0, x1 = box
    return data[:, z0:z1, y0:y1, x0:x1]


def anti_aliasing_sigmas(
    input_size: tuple[int, ...], output_size: tuple[int, ...]
) -> tuple[float, ...]:
    """
    Gaussian sigmas (in input voxels) per axis that remove the detail finer than the output
    voxels before resizing, as 'skimage.transform.resize(anti_aliasing=True)': (factor - 1) / 2
    for the axes that shrink by 'factor', 0 for those that keep their size or grow.
    """
    return tuple(max(0.0, (i / o - 1) / 2) for i, o in zip(input_size, output_size))


def resize_with_anti_aliasing(
    image: torch.Tensor, output_size: tuple[int, int, int]
) -> torch.Tensor:
    """Resize (b, c, z, y, x) trilinearly, after a Gaussian filter along the shrinking axes."""
    sigmas = anti_aliasing_sigmas(tuple(image.shape[-3:]), tuple(output_size))
    if any(sigma > 0 for sigma in sigmas):
        filtered = ndi.gaussian_filter(
            image.numpy(), sigma=(0, 0, *sigmas), mode="nearest", truncate=4.0
        )
        image = torch.from_numpy(filtered)
    if tuple(image.shape[-3:]) == tuple(output_size):
        return image
    return F.interpolate(image, size=output_size, mode="trilinear", align_corners=False)


def min_roi_voxels(
    input_shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float] | None,
) -> tuple[int, int, int]:
    """
    The smallest ROI (z, y, x) in voxels of 'spacing' for 'enforce_min_bbox' of
    'resized_direct': the input shape at the target spacing, in mm (at the volume's own spacing
    without one), the extent 'resized' enforces after resampling.
    """
    if target_spacing is None:
        return tuple(input_shape)
    return tuple(
        math.ceil(size * target / current - 1e-6)
        for size, target, current in zip(input_shape, target_spacing, spacing)
    )


def fit_to_input_shape(
    roi: np.ndarray,
    normalization: str,
    processing_mode: str,
    input_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Normalize the (c, z, y, x) ROI again (z_score, min_max; HU windows are already applied) and
    resize or center crop/pad it to (1, c, *input_shape).
    """
    if normalization not in HU_WINDOWS:
        roi = apply_normalization(roi, normalization)
    image = torch.from_numpy(np.asarray(roi, dtype=np.float32)[None])
    if processing_mode == "center_crop":
        return center_crop_or_pad_3d(image, input_shape)
    if processing_mode == "resized":
        if image.shape[-3:] == tuple(input_shape):
            return image
        return F.interpolate(image, size=input_shape, mode="trilinear", align_corners=False)
    if processing_mode == "resized_direct":
        return resize_with_anti_aliasing(image, input_shape)
    raise ValueError(f"Unknown processing mode: {processing_mode}")


def preprocess_encoder_input(
    image: sitk.Image,
    mask: Optional[sitk.Image],
    *,
    target_spacing: Optional[Tuple[float, float, float]],
    normalization: str,
    input_shape: Tuple[int, int, int],
    processing_mode: str,
    mask_label_ids: Optional[List[int]] = None,
    enforce_min_bbox: bool = True,
    outlier_method: str = "none",
    outlier_min_voxels: int = 400,
) -> torch.Tensor:
    """
    Single-block encoder input (1, 1, *input_shape) of a volume, cropped to the mask ROI if a mask
    is given. Spacings are (z, y, x), matching the arrays of SimpleITK; without 'target_spacing',
    the volume keeps its spacing.
    """
    data = sitk.GetArrayFromImage(image)[np.newaxis]
    data = apply_normalization(data, normalization)
    if processing_mode == "resized_direct":
        # Cropped at the volume's own spacing and resized once, in 'fit_to_input_shape'.
        if mask is not None:
            mask_data = binarize_mask(
                sitk.GetArrayFromImage(mask)[np.newaxis],
                mask_label_ids,
                outlier_method,
                outlier_min_voxels,
            )
            if mask_data.shape != data.shape:
                raise ValueError(
                    f"The mask has shape {mask_data.shape[1:]}, the volume {data.shape[1:]}; "
                    "'resized_direct' needs masks on the grid of their volumes."
                )
            min_size_zyx = min_roi_voxels(input_shape, spacing_zyx(image), target_spacing)
            data = crop_to_mask_roi(data, mask_data, min_size_zyx if enforce_min_bbox else None)
        return fit_to_input_shape(data, normalization, processing_mode, input_shape)
    spacing = spacing_zyx(image)
    mask_data = None
    if mask is not None:
        mask_data = binarize_mask(
            sitk.GetArrayFromImage(mask)[np.newaxis],
            mask_label_ids,
            outlier_method,
            outlier_min_voxels,
        )
    if target_spacing is not None:
        # z_score and min_max are computed on the whole ROI, so only HU windows allow the crop.
        crop_to = input_shape
        if processing_mode != "center_crop" or normalization not in HU_WINDOWS:
            crop_to = None
        roi = resampled_roi(
            data,
            mask_data,
            spacing,
            spacing_zyx(mask) if mask is not None else spacing,
            target_spacing,
            input_shape if enforce_min_bbox and mask is not None else None,
            crop_to,
        )
        if roi is not None:
            return fit_to_input_shape(roi, normalization, processing_mode, input_shape)
        data = resample_to_spacing(data, spacing, target_spacing)
    if mask_data is not None:
        if target_spacing is not None:
            mask_data = resample_to_spacing(
                mask_data, spacing_zyx(mask), target_spacing, is_seg=True
            )
        data = crop_to_mask_roi(data, mask_data, input_shape if enforce_min_bbox else None)
    return fit_to_input_shape(data, normalization, processing_mode, input_shape)
