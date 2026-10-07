"""Image volume preprocessing and training augmentation."""

from __future__ import annotations

import numpy as np
import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import (
    BGContrast,
    ContrastTransform,
)
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform


# from nnUNetv2
@staticmethod
def get_training_image_transforms(patch_size) -> BasicTransform:
    transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    # not too much zoom because otherwise not everything visible from lung
    # the rest is nnunet default
    transforms.append(
        SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
            p_rotation=0.2, p_scaling=0.2, scaling=(0.95, 1.05), p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False  # , mode_seg='nearest'
        )
    )

    transforms.append(RandomTransform(
        GaussianNoiseTransform(
            noise_variance=(0, 0.1),
            p_per_channel=1,
            synchronize_channels=True
        ), apply_probability=0.1
    ))
    transforms.append(RandomTransform(
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.),
            synchronize_channels=False,
            synchronize_axes=False,
            p_per_channel=0.5, benchmark=True
        ), apply_probability=0.2
    ))
    transforms.append(RandomTransform(
        MultiplicativeBrightnessTransform(
            multiplier_range=BGContrast((0.75, 1.25)),
            synchronize_channels=False,
            p_per_channel=1
        ), apply_probability=0.15
    ))
    transforms.append(RandomTransform(
        ContrastTransform(
            contrast_range=BGContrast((0.75, 1.25)),
            preserve_range=True,
            synchronize_channels=False,
            p_per_channel=1
        ), apply_probability=0.15
    ))
    transforms.append(RandomTransform(
        SimulateLowResolutionTransform(
            scale=(0.5, 1),
            synchronize_channels=False,
            synchronize_axes=True,
            ignore_axes=ignore_axes,
            allowed_channels=None,
            p_per_channel=0.5
        ), apply_probability=0.25
    ))
    transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=1,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ), apply_probability=0.1
    ))
    transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=0,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ), apply_probability=0.3
    ))

    return ComposeTransforms(transforms)


def preprocess_3d_image(image_array, do_resize, do_transforms, patch_size, split, training_transforms=None):
    """Preprocess 3D image: convert to float32, remove singleton dims, pad if needed, and random crop to final patch size"""
    # Ensure float32 dtype
    if not isinstance(image_array, np.ndarray):
        image_array = np.array(image_array, dtype=np.float32)
    else:
        image_array = image_array if image_array.dtype == np.float32 else image_array.astype(np.float32)

    # Remove singleton first dimension if present
    if image_array.ndim == 4 and image_array.shape[0] == 1:
        image_array = image_array.squeeze(0)

    # Get image dimensions (assuming format: H, W, D)
    h, w, d = image_array.shape

    # Pad if image is smaller than patch_size
    pad_h = max(0, patch_size[0] - h)
    pad_w = max(0, patch_size[1] - w)
    pad_d = max(0, patch_size[2] - d)

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        # Split padding evenly on both sides to center the image
        pad_h_before = pad_h // 2
        pad_h_after = pad_h - pad_h_before

        pad_w_before = pad_w // 2
        pad_w_after = pad_w - pad_w_before

        pad_d_before = pad_d // 2
        pad_d_after = pad_d - pad_d_before

        # Pad the image
        padding = ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (pad_d_before, pad_d_after))
        image_array = np.pad(image_array, padding, mode='constant', constant_values=0)

    if do_resize:

        # interpolate braucht 5D
        image_array = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
        image_array = torch.nn.functional.interpolate(image_array, size=patch_size, mode='trilinear', align_corners=False)

        if do_transforms:
            image_array = torch.squeeze(training_transforms(**{'image': image_array.squeeze(0)})['image'], dim=0)
        else:
            image_array = image_array.squeeze(0).squeeze(0)

    else:
        # Random crop to exact patch size (if larger than patch_size)
        h, w, d = image_array.shape
        if h > patch_size[0] or w > patch_size[1] or d > patch_size[2]:
            if split != "train":
                # Center crop
                start_h = (h - patch_size[0]) // 2
                start_w = (w - patch_size[1]) // 2
                start_d = (d - patch_size[2]) // 2
            else:
                # Random patch
                max_starts = [max(0, image_array.shape[i] - patch_size[i]) for i in range(3)]
                start_h, start_w, start_d = [int(np.clip(np.random.normal(m/2, m*0.1), 0, m)) for m in max_starts]

            # Extract patch
            image_array = image_array[
                start_h:start_h + patch_size[0],
                start_w:start_w + patch_size[1],
                start_d:start_d + patch_size[2]
            ]

    return image_array
