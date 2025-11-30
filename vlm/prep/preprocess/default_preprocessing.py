from pathlib import Path
import sys
import os

from tqdm import tqdm

from utils.crop_to_mask import (
    enforce_min_size,
    expand_bounding_box,
    get_mask_bounding_box,
)
from utils.default_resampling import (
    resample_data_or_seg_to_spacing,
)
import numpy as np
import multiprocessing as mp
import SimpleITK as sitk
from utils.blosc_helper import save_case, comp_blosc2_params
from batchgenerators.utilities.file_and_folder_operations import *
from utils.normalization import ZScoreNormalization
from typing import List, Optional


def run_case(params) -> None:
    """
    Preprocesses a 3D image case by resampling to target spacing, optionally using
    a segmentation mask for region-of-interest extraction, cropping around the mask
    bounding box, normalizing, and saving the result as compressed blocks.

    Parameters packed in `params`:
        image_base_path (str): Base directory containing input images.
        masks_base_path (Optional[str]): Base directory containing masks (same
            relative structure as images). If None, no mask is used.
        rel_img_path (str): Relative path of the image file (within image_base_path).
        target_spacing (Tuple[float, float, float]): Desired spacing for resampling.
        patch_size (Tuple[int, int, int]): Target patch size (used for chunk/block sizes).
        output_dir (str): Directory where output files will be saved.
        min_size_xyz (Optional[List[int]]): Minimum cropped size in voxels.
        apply_mask (bool): If True, multiply the image by the binary mask before cropping.
        mask_label_ids (Optional[List[int]]): Label IDs to treat as foreground in the
            mask. If None or empty, any non-zero label is treated as foreground.

    Notes:
        - If a mask is found, it is resampled to the target spacing and its bounding
          box is used to crop the image.
        - Cropping is performed if a valid mask exists, independent of apply_mask.
        - If no mask is present (or mask_folder is None), the full image is used.
    """
    (
        image_base_path,
        masks_base_path,
        rel_img_path,
        target_spacing,
        patch_size,
        output_dir,
        min_size_xyz,
        apply_mask,
        mask_label_ids,
    ) = params

    img_path = os.path.join(image_base_path, rel_img_path)

    if masks_base_path is not None:
        mask_path = os.path.join(masks_base_path, rel_img_path)
        has_mask = os.path.isfile(mask_path)
    else:
        mask_path = None
        has_mask = False

    out_path_truncated = os.path.join(output_dir, Path(rel_img_path).name.replace(".nii.gz", ""))

    # check if out_path_truncated.b2nd already exists
    if os.path.isfile(out_path_truncated + ".b2nd"):
        print(f"File {out_path_truncated + '.b2nd'} already exists, skipping.")
        return

    try:
        Path(out_path_truncated).parent.mkdir(parents=True, exist_ok=True)

        img = sitk.ReadImage(img_path)
        data = sitk.GetArrayFromImage(img)[np.newaxis, ...]  # (1, z, y, x)
        original_spacing = img.GetSpacing()[::-1]  # (x, y, z) -> (z, y, x)

        normalizer = ZScoreNormalization()
        data = normalizer.run(data)

        # Skip resampling if original spacing is already the target spacing
        if np.allclose(original_spacing, target_spacing, atol=1e-2):
            print(
                f"Image {rel_img_path} already has target spacing {target_spacing}, "
                f"skipping resampling."
            )
            data_1mm = data
        else:
            data_1mm = resample_data_or_seg_to_spacing(data, original_spacing, target_spacing)

        if has_mask:
            mask_img = sitk.ReadImage(mask_path)
            mask_data = sitk.GetArrayFromImage(mask_img)[np.newaxis, ...]  # (1, z, y, x)

            # Convert to binary mask using specified label IDs
            if mask_label_ids is None or len(mask_label_ids) == 0:
                # Treat any non-zero label as foreground
                mask_data = (mask_data > 0).astype(np.uint8)
            else:
                mask_data = np.isin(mask_data, mask_label_ids).astype(np.uint8)

            original_spacing_mask = mask_img.GetSpacing()[::-1]  # (z, y, x)
            if np.allclose(original_spacing_mask, target_spacing, atol=1e-2):
                print(
                    f"Mask {rel_img_path} already has target spacing {target_spacing}, "
                    f"skipping resampling."
                )
                mask_1mm = mask_data
            else:
                mask_1mm = resample_data_or_seg_to_spacing(
                    mask_data, original_spacing_mask, target_spacing, is_seg=True
                )

            # Extract bounding box from the mask
            mask_bbox = get_mask_bounding_box(mask_1mm[0])
            # Expand bbox to give some margin
            mask_bbox = expand_bounding_box(
                mask_bbox,
                data_1mm.shape[3],  # x-dim
                data_1mm.shape[2],  # y-dim
                data_1mm.shape[1],  # z-dim
                0.04,
            )

            # Ensure the mask bbox is at least min_size_xyz if provided
            if min_size_xyz:
                mask_bbox = enforce_min_size(mask_bbox, min_size_xyz, data_1mm.shape[1:])

            z0, z1, y0, y1, x0, x1 = mask_bbox
            resized_mask = mask_1mm[:, z0:z1, y0:y1, x0:x1]

            if apply_mask:
                # Zero-out everything outside the (resampled) full mask
                data_1mm *= mask_1mm

            resized_img = data_1mm[:, z0:z1, y0:y1, x0:x1]
        else:
            resized_img = data_1mm

        block_size_data, chunk_size_data = comp_blosc2_params(
            resized_img.shape, patch_size, resized_img.itemsize
        )

        save_case(
            resized_img,
            out_path_truncated,
            chunks=chunk_size_data,
            blocks=block_size_data,
        )
    except Exception as e:
        print(f"Error processing {rel_img_path}")  # intentionally not printing e to avoid spam


def load_crop_mask_normalize_images(
    in_folder: str,
    mask_folder: Optional[str],
    out_folder: str,
    target_spacing: List[float],
    patch_size: List[int],
    min_size_xyz: Optional[List[int]] = None,
    apply_mask: bool = False,
    num_workers: int = 1,
    mask_label_ids: Optional[List[int]] = None,
) -> None:
    """
    Preprocess all images in `in_folder` by:

      - resampling to target spacing,
      - optionally using masks from `mask_folder` for ROI extraction,
      - cropping around the mask bounding box (if mask exists),
      - z-score normalization,
      - saving results as compressed blocks.

    Parameters:
        in_folder (str): Path to input directory containing images (*.nii.gz).
        mask_folder (Optional[str]): Path to directory containing masks. If None,
            no masks are used. Relative paths must match those in in_folder.
        out_folder (str): Directory where processed images are saved.
        target_spacing (List[float]): Desired spacing [z, y, x] in mm.
        patch_size (List[int]): Target patch size (used for Blosc2 chunk/block config).
        min_size_xyz (Optional[List[int]]): Minimum cropped region size [z, y, x].
        apply_mask (bool): If True, multiply the image by the binary mask before cropping.
        num_workers (int): Number of parallel worker processes.
        mask_label_ids (Optional[List[int]]): Label IDs treated as foreground in masks.
    """
    os.makedirs(out_folder, exist_ok=True)
    all_images = list(Path(in_folder).rglob("*.nii.gz"))

    tasks = [
        (
            in_folder,
            mask_folder,
            img_path.relative_to(in_folder).as_posix(),
            target_spacing,
            patch_size,
            out_folder,
            min_size_xyz,
            apply_mask,
            mask_label_ids,
        )
        for img_path in all_images
    ]

    with (
        mp.Pool(processes=num_workers) as pool,
        tqdm(total=len(tasks), desc="Processing", unit="file") as pbar,
    ):
        for _ in pool.imap_unordered(run_case, tasks):
            pbar.update()


def create_label_dict(path):
    label_dict = load_json(path)
    return label_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "in_folder",
        type=str,
        help="Path to input directory containing images (e.g. 'imagesTr').",
    )
    parser.add_argument(
        "mask_folder",
        type=str,
        help="Path to directory containing masks. Use 'None' to disable masks.",
    )
    parser.add_argument(
        "out_folder",
        type=str,
        help="Directory where the processed images will be saved.",
    )

    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Desired spacing [z, y, x] in mm (default: 1.0 1.0 1.0).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[192, 192, 192],
        help="Target patch size (default: 192 192 192).",
    )
    parser.add_argument(
        "--min_size_xyz",
        type=int,
        nargs=3,
        default=None,
        help="Minimum size of the cropped region in voxels (default: None).",
    )
    parser.add_argument(
        "--apply_mask",
        action="store_true",
        help="If set, multiply image by the binary mask before cropping (default: False).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers to use (default: 1).",
    )
    parser.add_argument(
        "--mask_label_ids",
        type=int,
        nargs="+",
        default=[10, 11, 12, 13, 14],
        help=(
            "Label IDs used as foreground in the mask; voxels with these labels are "
            "set to 1, others to 0 (default: 10 11 12 13 14). "
            "If you pass an empty list in code, any non-zero label is treated as foreground."
        ),
    )

    args = parser.parse_args()

    # Interpret literal "None" (any case) as no mask folder
    if args.mask_folder.lower() == "none":
        mask_folder = None
    else:
        mask_folder = args.mask_folder

    load_crop_mask_normalize_images(
        args.in_folder,
        mask_folder,
        args.out_folder,
        args.target_spacing,
        args.patch_size,
        min_size_xyz=args.min_size_xyz,
        apply_mask=args.apply_mask,
        num_workers=args.num_workers,
        mask_label_ids=args.mask_label_ids,
    )
