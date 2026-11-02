"""
NIfTI volumes as vision encoder inputs, for inference on new images ('inference_nifti.py') and
for preparing the images of image-mode training.
"""

from pathlib import Path

import SimpleITK as sitk
from torch.utils.data import Dataset

from vlm.data.preprocessing.encoder_input import preprocess_encoder_input
from vlm.utils import logger

NIFTI_SUFFIXES = (".nii.gz", ".nii")


def strip_nifti_suffix(path: Path) -> str:
    return next(str(path)[: -len(s)] for s in NIFTI_SUFFIXES if str(path).endswith(s))


def find_nifti_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.name.endswith(NIFTI_SUFFIXES)
    )


class NiftiEncoderInputDataset(Dataset):
    """
    Reorients and preprocesses NIfTI volumes into encoder inputs (1, *input_shape), with the
    ROI mask of the same relative path in 'mask_dir', if given. Unreadable volumes give a sample
    with an 'error' instead of an 'encoder_input'.
    """

    def __init__(
        self,
        input_dir: Path,
        files: list[Path],
        orientation: str,
        mask_dir: Path | None,
        preprocessing_kwargs: dict,
    ):
        self.input_dir = input_dir
        self.files = files
        self.orientation = orientation
        self.mask_dir = mask_dir
        self.preprocessing_kwargs = preprocessing_kwargs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        relative_path = path.relative_to(self.input_dir)
        sample = {"image_id": strip_nifti_suffix(relative_path), "path": str(path)}
        try:
            image = sitk.DICOMOrient(sitk.ReadImage(path), self.orientation)
            mask = None
            if self.mask_dir is not None:
                mask_path = self.mask_dir / relative_path
                if mask_path.is_file():
                    mask = sitk.DICOMOrient(sitk.ReadImage(mask_path), self.orientation)
                else:
                    logger.warning(f"Mask {mask_path} not found, using the whole image.")
            encoder_input = preprocess_encoder_input(image, mask, **self.preprocessing_kwargs)
            return {**sample, "encoder_input": encoder_input[0], "error": None}
        except Exception as error:  # Skip unreadable volumes instead of stopping the run.
            return {**sample, "encoder_input": None, "error": f"{type(error).__name__}: {error}"}


def collate_samples(samples: list[dict]) -> list[dict]:
    return samples
