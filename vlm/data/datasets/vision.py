"""Loading and preprocessing of image volumes and precomputed embeddings."""

import os
from pathlib import Path
from typing import Callable

import blosc2
import nrrd
import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange

from vlm.data.types import VisionInputType


def load_vision_input(
    vision_input_path: Path,
    *,
    vision_input_type: VisionInputType,
    is_stacked_2d_embedding: bool,
    reduce_to_n_slices: int | None,
    patch_size: tuple[int, int, int] | None = None,
    image_transforms: Callable | None = None,
):
    """
    Load a vision input from file: an embedding, or, in image mode, a preprocessed encoder input
    of shape 'patch_size' (as made by 'preprocess_encoder_input'), augmented with
    'image_transforms' if given.
    """
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
        return prepare_image_input(vision_input, vision_input_path, patch_size, image_transforms)

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


def prepare_image_input(
    image: np.ndarray,
    path: Path,
    patch_size: tuple[int, int, int],
    image_transforms: Callable | None,
) -> np.ndarray:
    """
    The (1, *patch_size) float32 encoder input of a preprocessed image, augmented with
    'image_transforms' if given. Images are not normalized, resized or cropped here.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1:] != tuple(patch_size):
        raise ValueError(
            f"Image {path} has shape {image.shape}, but image mode expects encoder inputs of "
            f"shape {tuple(patch_size)} or (1, *{tuple(patch_size)}). Preprocess the images "
            "into encoder inputs before training, e.g. with "
            "'python -m vlm.data.preprocessing.prepare_image_inputs'."
        )
    if image_transforms is not None:
        image = image_transforms(image=torch.from_numpy(image))["image"].numpy()
    return image
