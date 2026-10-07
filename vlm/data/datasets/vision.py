"""Loading and preprocessing of image volumes and precomputed embeddings."""

import os
from numbers import Number
from pathlib import Path
from typing import Type

import blosc2
import nrrd
import numpy as np
import SimpleITK as sitk
from einops import rearrange

from vlm.utils.data.dataset_utils import preprocess_3d_image
from vlm.utils.data.vision_input_type import VisionInputType


def convert_dtype(image: np.ndarray, target_dtype: Type[Number]) -> np.ndarray:
    return image.astype(target_dtype)


def z_score_normalization(
    image: np.ndarray,
    use_mask_for_norm: bool,
    non_zero_mask: np.ndarray,
    target_dtype: Type[Number],
) -> np.ndarray:
    image = convert_dtype(image, target_dtype)
    if use_mask_for_norm:
        mask = non_zero_mask >= 0
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / max(std, 1e-8)
    else:
        mean = image.mean()
        std = image.std()
        image = (image - mean) / max(std, 1e-8)
    return image


def load_vision_input(
    vision_input_path: Path,
    *,
    vision_input_type: VisionInputType,
    do_resize: bool,
    do_transforms: bool,
    patch_size: tuple[int, ...] | None,
    split: str,
    training_transforms,
    is_stacked_2d_embedding: bool,
    reduce_to_n_slices: int | None,
):
    """Load vision input from file"""
    if str(vision_input_path).endswith(".npy"):
        vision_input = np.load(vision_input_path)
    elif str(vision_input_path).endswith(".npz"):
        # Use context manager to ensure proper file closure
        with np.load(vision_input_path) as npz_file:
            # Attempt to load the "embeddings" key if it exists, otherwise load the first array
            if "embedding" in npz_file:
                vision_input = npz_file[
                    "embeddings"
                ].copy()  # Make a copy to avoid reference issues
            else:
                vision_input = npz_file[
                    npz_file.files[0]
                ].copy()  # Make a copy to avoid reference issues

    elif str(vision_input_path).endswith(".b2nd"):
        # Load b2nd file like nnUNet :)
        dparams = {"nthreads": 1}
        mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}

        vision_input = np.array(
            blosc2.open(urlpath=vision_input_path, mode="r", dparams=dparams, **mmap_kwargs)
        )

    elif str(vision_input_path).endswith(".bl2"):
        # remove batch dimension
        vision_input = blosc2.load_array(vision_input_path)

        if vision_input.ndim == 5 and vision_input.shape[0] == 1:
            vision_input = vision_input.squeeze(0)

    elif str(vision_input_path).endswith(".nrrd"):
        vision_input, header = nrrd.read(vision_input_path)

    elif str(vision_input_path).endswith((".nii.gz", ".nii")):
        vision_input = sitk.GetArrayFromImage(sitk.ReadImage(vision_input_path))

    else:
        raise ValueError(f"Unsupported vision input file format: {vision_input_path}")

    if vision_input_type == VisionInputType.IMAGE:
        vision_input = z_score_normalization(vision_input, False, None, np.float32)
        vision_input = preprocess_3d_image(
            image_array=vision_input,
            do_resize=do_resize,
            do_transforms=do_transforms,
            patch_size=patch_size,
            split=split,
            training_transforms=training_transforms,
        )

    if is_stacked_2d_embedding:
        # remove class token
        vision_input = vision_input[:, 1:, :]

        n_slices, n_patch_tokens, embed_dim = vision_input.shape
        n_side = int(n_patch_tokens ** 0.5)

        vision_input = rearrange(vision_input, "s (h w) c -> s h w c", h=n_side, w=n_side)

        bins = np.linspace(0, n_slices, reduce_to_n_slices + 1).astype(int)

        # do 'average pooling' to compress slice dimension
        pooled_slices = [
            vision_input[bins[i]:bins[i + 1]].mean(axis=0)
            for i in range(reduce_to_n_slices)
        ]
        vision_input = np.stack(pooled_slices, axis=0)
        vision_input = rearrange(vision_input, "s h w c -> c s h w")

    elif vision_input.ndim == 3:
        # add channel dim if missing
        vision_input = vision_input[None, ...]

    return vision_input
