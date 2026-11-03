"""
Preprocessing of CT volumes into vision encoder inputs, shared by the embedding generation
(vlm/scripts/embeddings/generate_embeddings.py) and the inference on NIfTI images.

Single-block pipeline per volume: normalize → resample to the encoder spacing, if it has one →
crop to the mask ROI (bounding box expanded by 4%) → normalize again → fit to the encoder input
shape by resizing ('resized') or center cropping/padding ('center_crop').
"""

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
    ANISO_THRESHOLD,
    compute_new_shape,
    determine_do_sep_z_and_axis,
    resample_data_or_seg,
)

PROCESSING_MODES = ("resized", "center_crop")
NORMALIZATIONS = ("z_score", "min_max", "vista3d_scale", "merlin_scale", "none")
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


def resample_data_or_seg_to_spacing(
    data: np.ndarray,
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Optional[bool] = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
) -> np.ndarray:
    """Resample data to new spacing with proper anisotropy handling."""
    do_separate_z, axis = determine_do_sep_z_and_axis(
        force_separate_z,
        current_spacing,
        new_spacing,
        separate_z_anisotropy_threshold,
    )

    assert data.ndim == 4, "data must be (c, z, y, x)"

    new_shape = compute_new_shape(data.shape[1:], current_spacing, new_spacing)
    return resample_data_or_seg(
        data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z
    )


def resample_to_spacing_if_needed(
    data: np.ndarray,
    current_spacing: Tuple[float, ...],
    target_spacing: Tuple[float, ...],
    is_seg: bool = False,
) -> np.ndarray:
    """Resample (c, z, y, x) data, unless it already has the target spacing (within 0.01 mm)."""
    if np.allclose(current_spacing, target_spacing, atol=1e-2):
        return data
    return resample_data_or_seg_to_spacing(data, current_spacing, target_spacing, is_seg=is_seg)


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

    elif scheme == "vista3d_scale":
        a_min = -963.8247715525971
        a_max = 1053.678477684517
        image = (image - a_min) / (a_max - a_min)
        image = np.clip(image, 0.0, 1.0)
        return image

    elif scheme == "merlin_scale":
        a_min = -1000.0
        a_max = 1000.0
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


def crop_to_mask_roi(
    data: np.ndarray, mask: np.ndarray, min_size_xyz: Optional[Tuple[int, int, int]]
) -> np.ndarray:
    """
    Crop (c, z, y, x) data to the mask bounding box, expanded by 4% and, with 'min_size_xyz',
    enlarged to at least that size where the image allows it.
    """
    box = get_mask_bounding_box(mask[0])
    box = expand_bounding_box(box, data.shape[3], data.shape[2], data.shape[1], 0.04)
    if min_size_xyz:
        box = enforce_min_size(box, min_size_xyz, data.shape[1:])
    z0, z1, y0, y1, x0, x1 = box
    return data[:, z0:z1, y0:y1, x0:x1]


def fit_to_input_shape(
    roi: np.ndarray,
    normalization: str,
    processing_mode: str,
    input_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Normalize the (c, z, y, x) ROI again and resize or center crop/pad it to (1, c, *input_shape)."""
    image = torch.from_numpy(apply_normalization(roi[None], normalization))
    if processing_mode == "center_crop":
        return center_crop_or_pad_3d(image, input_shape)
    if processing_mode == "resized":
        if image.shape[-3:] == tuple(input_shape):
            return image
        return F.interpolate(image, size=input_shape, mode="trilinear", align_corners=False)
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
    if target_spacing is not None:
        data = resample_to_spacing_if_needed(data, image.GetSpacing()[::-1], target_spacing)
    if mask is not None:
        mask_data = binarize_mask(
            sitk.GetArrayFromImage(mask)[np.newaxis],
            mask_label_ids,
            outlier_method,
            outlier_min_voxels,
        )
        if target_spacing is not None:
            mask_data = resample_to_spacing_if_needed(
                mask_data, mask.GetSpacing()[::-1], target_spacing, is_seg=True
            )
        data = crop_to_mask_roi(data, mask_data, input_shape if enforce_min_bbox else None)
    return fit_to_input_shape(data, normalization, processing_mode, input_shape)
