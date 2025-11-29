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
from typing import List, Optional



def run_case(params) -> None:
    """
    Preprocesses a multi-modal medical image case by resampling to target spacing,
    putting images into a canonical orientation, scaling HU by 1/1000 and clipping
    to [-1, 1], optionally applying lung extraction and cropping, and saving as
    compressed blosc2 blocks.
    """
    (
        image_base_path,
        masks_base_path,
        rel_img_path,
        target_spacing,
        patch_size,
        output_dir,
        min_size_xyz,
        lung_extract,
    ) = params
    img_path = os.path.join(image_base_path, rel_img_path)
    mask_path = os.path.join(masks_base_path, rel_img_path)
    has_mask = os.path.isfile(mask_path)

    out_path_truncated = os.path.join(output_dir, Path(rel_img_path).name.replace(".nii.gz", ""))

    # check if out_path_truncated.b2nd already exists
    if os.path.isfile(out_path_truncated + ".b2nd"):
        print(f"File {out_path_truncated + '.b2nd'} already exists, skipping.")
        return

    try:
        Path(out_path_truncated).parent.mkdir(parents=True, exist_ok=True)

        # --- 1) load and put image into canonical orientation ---
        img = sitk.ReadImage(img_path)
        # choose canonical orientation; LPS is a common choice
        img = sitk.DICOMOrient(img, "LPS")

        # SimpleITK arrays are always (z, y, x); we add channel dim -> (C, Z, Y, X)
        data = sitk.GetArrayFromImage(img)[np.newaxis, ...].astype(np.float32)
        original_spacing = img.GetSpacing()[::-1]  # (x,y,z) -> (z,y,x)

        # --- 2) resample to target spacing (like tio.Resample(spacing, ...)) ---
        if np.allclose(original_spacing, target_spacing, atol=1e-2):
            print(
                f"Image {rel_img_path} already has target spacing {target_spacing}, skipping resampling."
            )
            data_1mm = data
        else:
            data_1mm = resample_data_or_seg_to_spacing(
                data, original_spacing, target_spacing
            ).astype(np.float32)

        # --- 3) intensity scaling: /1000 then clamp to [-1, 1] ---
        # mimics: /1000; then tio.Clamp(*in_min_max) with in_min_max=(-1, 1)
        data_1mm /= 1000.0
        np.clip(data_1mm, -1.0, 1.0, out=data_1mm)

        if has_mask:
            # --- 4) load mask, canonical orient, resample, make lung-only binary mask ---
            mask_img = sitk.ReadImage(mask_path)
            mask_img = sitk.DICOMOrient(mask_img, "LPS")
            mask_data = sitk.GetArrayFromImage(mask_img)[np.newaxis, ...].astype(np.int16)

            # keep only lung labels [10, 11, 12, 13, 14]
            mask_data = np.isin(mask_data, [10, 11, 12, 13, 14]).astype(np.uint8)

            mask_spacing = mask_img.GetSpacing()[::-1]  # (x,y,z) -> (z,y,x)
            if np.allclose(mask_spacing, target_spacing, atol=1e-2):
                print(
                    f"Mask {rel_img_path} already has target spacing {target_spacing}, skipping resampling."
                )
                mask_1mm = mask_data
            else:
                mask_1mm = resample_data_or_seg_to_spacing(
                    mask_data, mask_spacing, target_spacing, is_seg=True
                ).astype(np.uint8)

            # --- 5) compute bbox on mask, maybe expand/enforce minimum size ---
            mask_bbox = get_mask_bounding_box(mask_1mm[0])
            mask_bbox = expand_bounding_box(
                mask_bbox,
                data_1mm.shape[3],  # X dim
                data_1mm.shape[2],  # Y dim
                data_1mm.shape[1],  # Z dim
                0.04,
            )
            if min_size_xyz:
                mask_bbox = enforce_min_size(mask_bbox, min_size_xyz, data_1mm.shape[1:])

            z0, z1, y0, y1, x0, x1 = mask_bbox
            resized_mask = mask_1mm[:, z0:z1, y0:y1, x0:x1]

            if lung_extract:
                data_1mm *= mask_1mm  # zero out outside lung

            resized_img = data_1mm[:, z0:z1, y0:y1, x0:x1]

        else:
            resized_img = data_1mm

        # --- 6) blosc2 chunk/blocking based on (possibly cropped) image and desired patch size ---
        block_size_data, chunk_size_data = comp_blosc2_params(
            resized_img.shape, patch_size, resized_img.itemsize
        )

        save_case(
            resized_img, out_path_truncated, chunks=chunk_size_data, blocks=block_size_data
        )

    except Exception as e:
        print(f"Error processing {rel_img_path}")  # : {e}


def load_crop_lungextract_normalize_images(
    in_folder: str,
    mask_folder: str,
    out_folder: str,
    target_spacing: List[float],
    patch_size: List[int],
    min_size_xyz: List[int] = None,
    lung_extract: bool = False,
    num_workers: int = 1,
) -> None:
    """
    Preprocesses all multi-modal lung images in a dataset by performing:
      - consistency checks for missing modalities,
      - resampling to target spacing,
      - optional skull stripping (lung extraction),
      - cropping or resizing to a fixed patch size,
      - z-score normalization (possibly using lung mask),
      - and saving the results as compressed blocks.

    A cross-validation split file is also generated.

    Parameters:
        in_folder (str): Path to input directory containing 'imagesTr' and optionally 'masks'.
        out_folder (str): Directory where the processed images and split file will be saved.
        target_spacing (List[float]): The desired spacing in mm.
        patch_size (List[int]): The target output patch size.
        lung_extract (bool, optional): Whether to apply skull stripping if a lung mask exists. Default is True.
        num_workers (int, optional): Number of parallel workers to use. Default is 1.

    Raises:
        RuntimeError: If some image identifiers are missing one or more expected modalities.
    """
    os.makedirs(out_folder, exist_ok=True)
    all_images = Path(in_folder).rglob("*.nii.gz")

    # Prepare args for parallel case processing
    tasks = [
        (
            in_folder,
            mask_folder,
            img_path.relative_to(in_folder).as_posix(),
            target_spacing,
            patch_size,
            out_folder,
            min_size_xyz,
            lung_extract,
        )
        for img_path in all_images
    ]

    with mp.Pool(processes=num_workers) as pool, tqdm(
        total=len(tasks), desc="Processing", unit="file"
    ) as pbar:
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
        help="Path to input directory containing 'imagesTr' and optionally 'masks'.",
    )
    parser.add_argument(
        "mask_folder",
        type=str,
        help="Path to directory containing lung masks.",
    )
    parser.add_argument(
        "out_folder",
        type=str,
        help="Directory where the processed images and split file will be saved.",
    )

    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Desired spacing in mm (default: [1.0, 1.0, 1.0]).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[160, 160, 160],
        help="Target output patch size (default: [160, 160, 160]).",
    )
    parser.add_argument(
        "--min_size_xyz",
        type=int,
        nargs=3,
        default=None,
        help="Minimum size of the cropped region in voxels (default: None).",
    )

    parser.add_argument(
        "--lung_extract",
        action="store_true",
        help="Whether to apply lung extraction using the lung mask (default: False).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers to use (default: 1).",
    )

    args = parser.parse_args()

    load_crop_lungextract_normalize_images(
        args.in_folder,
        args.mask_folder,
        args.out_folder,
        args.target_spacing,
        args.patch_size,
        min_size_xyz=args.min_size_xyz,
        lung_extract=args.lung_extract,
        num_workers=args.num_workers,
    )
