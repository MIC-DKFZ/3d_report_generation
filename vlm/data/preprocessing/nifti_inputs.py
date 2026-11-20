"""
NIfTI volumes as vision encoder inputs, for inference on new images ('inference_nifti.py') and
for preparing the images of image-mode training.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

from vlm.data.preprocessing.encoder_input import preprocess_encoder_input
from vlm.utils import logger

NIFTI_SUFFIXES = (".nii.gz", ".nii")


def strip_nifti_suffix(path: Path) -> str:
    return next(str(path)[: -len(s)] for s in NIFTI_SUFFIXES if str(path).endswith(s))


class EmptyMaskError(ValueError):
    """A mask without a voxel of the ROI."""


def check_mask(image: sitk.Image, mask: sitk.Image, mask_label_ids: list[int] | None) -> None:
    """
    Raise if a mask is not on the grid of its volume, or has no voxel of the ROI
    ('EmptyMaskError'): cropping to it would then give another input than in training.
    """
    for name, a, b, tolerance in (
        ("size", image.GetSize(), mask.GetSize(), 0),
        ("spacing", image.GetSpacing(), mask.GetSpacing(), 1e-3),
        ("origin", image.GetOrigin(), mask.GetOrigin(), 1e-2),
        ("direction", image.GetDirection(), mask.GetDirection(), 1e-3),
    ):
        if not np.allclose(a, b, rtol=0, atol=tolerance):
            raise ValueError(f"The mask's {name} {b} differs from the volume's {a}.")
    labels = sitk.GetArrayViewFromImage(mask)
    in_roi = labels > 0 if mask_label_ids is None else np.isin(labels, mask_label_ids)
    if not in_roi.any():
        raise EmptyMaskError(
            "The mask has no voxel of the ROI"
            + ("." if mask_label_ids is None else f" (labels {mask_label_ids}).")
        )


def find_nifti_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.name.endswith(NIFTI_SUFFIXES)
    )


class NiftiEncoderInputDataset(Dataset):
    """
    Reorients and preprocesses NIfTI volumes into encoder inputs (1, *input_shape), cropped to
    the ROI mask of the same relative path in 'mask_dir', if given. Unreadable volumes give a
    sample with an 'error' instead of an 'encoder_input'.

    A volume without a mask is used whole, as for the few training volumes without one; with
    'strict_masks', for inference on new volumes, it gives an error instead, as do masks that are
    not on the grid of their volume or have no voxel of the ROI. 'allow_missing_masks' uses the
    volumes without a mask, or with an empty one, whole even then; their samples are marked with
    'whole_volume'. A mask off the grid of its volume stays an error.

    'preprocess' replaces 'preprocess_encoder_input': it maps the reoriented volume and mask to
    the sample's fields, e.g. the blocks of a multi-block input.
    """

    def __init__(
        self,
        input_dir: Path,
        files: list[Path],
        orientation: str,
        mask_dir: Path | None,
        preprocessing_kwargs: dict,
        strict_masks: bool = False,
        allow_missing_masks: bool = False,
        preprocess: Callable[[sitk.Image, sitk.Image | None], dict] | None = None,
    ):
        self.input_dir = input_dir
        self.files = files
        self.orientation = orientation
        self.mask_dir = mask_dir
        self.preprocessing_kwargs = preprocessing_kwargs
        self.strict_masks = strict_masks
        self.allow_missing_masks = allow_missing_masks
        self.preprocess = preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        relative_path = path.relative_to(self.input_dir)
        sample = {"image_id": strip_nifti_suffix(relative_path), "path": str(path)}
        try:
            image = sitk.ReadImage(path)
            mask, whole_volume = None, False
            if self.mask_dir is not None:
                mask_path = self.mask_dir / relative_path
                if mask_path.is_file():
                    mask = sitk.ReadImage(mask_path)
                    if self.strict_masks:
                        try:
                            check_mask(
                                image, mask, self.preprocessing_kwargs.get("mask_label_ids")
                            )
                        except EmptyMaskError as error:
                            if not self.allow_missing_masks:
                                raise
                            logger.warning(f"Mask {mask_path}: {error} Using the whole volume.")
                            mask, whole_volume = None, True
                    if mask is not None:
                        mask = sitk.DICOMOrient(mask, self.orientation)
                elif self.strict_masks and not self.allow_missing_masks:
                    raise FileNotFoundError(f"Mask {mask_path} not found.")
                else:
                    logger.warning(f"Mask {mask_path} not found, using the whole volume.")
                    whole_volume = True
            image = sitk.DICOMOrient(image, self.orientation)
            if self.preprocess is not None:
                inputs = self.preprocess(image, mask)
            else:
                inputs = {
                    "encoder_input": preprocess_encoder_input(
                        image, mask, **self.preprocessing_kwargs
                    )[0]
                }
            return {**sample, **inputs, "whole_volume": whole_volume, "error": None}
        except Exception as error:  # Skip unreadable volumes instead of stopping the run.
            return {**sample, "encoder_input": None, "error": f"{type(error).__name__}: {error}"}


def collate_samples(samples: list[dict]) -> list[dict]:
    return samples
