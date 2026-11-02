"""Training augmentation of image-mode encoder inputs."""

from __future__ import annotations

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
def get_training_image_transforms(patch_size) -> BasicTransform:
    """nnU-Net augmentation of (C, *patch_size) inputs, with milder scaling and no elastic deformation."""
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
