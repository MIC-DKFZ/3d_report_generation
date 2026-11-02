"""
Preprocess NIfTI volumes into the encoder inputs of image-mode training.

Image-mode training uses its inputs as stored (it only augments them), so the volumes are
preprocessed here, with the same function as the embeddings and the inference on new volumes
('preprocess_encoder_input'): reorient → normalize → resample → crop to the ROI mask → resize or
center crop/pad to the encoder input shape. Each volume '<name>.nii.gz' found (recursively) in
'--input_dir' is saved as '<vision_input_path>/<split_folder>/<name>.b2nd' of shape
(1, *input_shape), the file the datasets load for the image ID '<name>'. Run the script once per
split folder of the dataset's layout (CT-RATE: 'train_fixed' and 'valid_fixed'; Merlin: none).

The settings are saved once per dataset, as '<vision_input_path>/vision_input_preprocessing.yml';
training copies them into the run config, so that inference on new volumes preprocesses them the
same way. All split folders must use the same settings.

    uv run python -m vlm.data.preprocessing.prepare_image_inputs --input_dir <volumes> \\
        --vision_input_path <vision_input_path> --split_folder train_fixed \\
        --mask_dir <masks> --roi_mask "<mask>" --orientation LPS --input_shape 192 192 192 \\
        --processing_mode resized
"""

import argparse
from pathlib import Path

import blosc2
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm.configs.data_config import VisionInputPreprocessingConfig, VisionInputSettings
from vlm.data.preprocessing.encoder_input import (
    NORMALIZATIONS,
    OUTLIER_METHODS,
    PROCESSING_MODES,
)
from vlm.data.preprocessing.nifti_inputs import (
    NiftiEncoderInputDataset,
    collate_samples,
    find_nifti_files,
)
from vlm.data.vision_input_settings import save_settings
from vlm.utils.serialization import save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument(
        "--vision_input_path",
        type=Path,
        required=True,
        help="Dataset folder, the 'data.vision_input_path' of the runs; holds the settings.",
    )
    parser.add_argument(
        "--split_folder",
        default="",
        help="Folder of the split in '--vision_input_path', e.g. 'train_fixed' for CT-RATE.",
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=None,
        help="ROI masks with the same relative paths as the volumes. Without, the whole volume.",
    )
    parser.add_argument(
        "--roi_mask",
        default=None,
        help="What the masks are, recorded in the settings (e.g. 'TotalSegmentator lungs'). "
        "Required with '--mask_dir'.",
    )
    parser.add_argument(
        "--orientation",
        required=True,
        help="Orientation the volumes are reoriented to, e.g. 'LPS' (see "
        "'data.vision_input_orientation').",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs=3,
        required=True,
        metavar=("Z", "Y", "X"),
        help="The vision encoder's input shape ('model.vision_encoder.patch_size').",
    )
    parser.add_argument("--processing_mode", choices=PROCESSING_MODES, required=True)
    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help="Resample to this spacing in mm; by default, the volumes keep their spacing.",
    )
    parser.add_argument("--normalization", choices=NORMALIZATIONS, default="z_score")
    parser.add_argument("--mask_label_ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--no_enforce_min_bbox",
        dest="enforce_min_bbox",
        action="store_false",
        help="Do not enlarge the ROI to at least the encoder input shape.",
    )
    parser.add_argument("--outlier_method", choices=OUTLIER_METHODS, default="none")
    parser.add_argument("--outlier_min_voxels", type=int, default=400)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--overwrite", action="store_true", help="Preprocess volumes already saved again."
    )
    args = parser.parse_args(argv)
    if (args.mask_dir is None) != (args.roi_mask is None):
        parser.error("'--mask_dir' and '--roi_mask' are needed together.")
    return args


def main(args: argparse.Namespace):
    settings = VisionInputSettings(
        vision_input_orientation=args.orientation,
        vision_input_preprocessing=VisionInputPreprocessingConfig(
            target_spacing=args.target_spacing,
            normalization=args.normalization,
            processing_mode=args.processing_mode,
            roi_mask=args.roi_mask,
            mask_label_ids=args.mask_label_ids,
            enforce_min_bbox=args.enforce_min_bbox,
            outlier_method=args.outlier_method,
            outlier_min_voxels=args.outlier_min_voxels,
        ),
    )
    output_dir = args.vision_input_path / args.split_folder
    input_dir = args.input_dir.resolve()
    files = find_nifti_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")
    output_paths = {
        file: output_dir / f"{Path(file.name.removesuffix('.gz')).stem}.b2nd" for file in files
    }
    duplicates = len(files) - len(set(output_paths.values()))
    if duplicates:
        raise ValueError(f"{duplicates} volumes in {input_dir} share a file name with another.")
    if not args.overwrite:
        files = [file for file in files if not output_paths[file].exists()]

    # Before preprocessing, so that settings differing from the dataset's fail early.
    settings_file = save_settings(args.vision_input_path, settings)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        NiftiEncoderInputDataset(
            input_dir,
            files,
            settings.vision_input_orientation,
            args.mask_dir,
            settings.vision_input_preprocessing.preprocessing_kwargs(args.input_shape),
        ),
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
    )
    failed = []
    for (sample,) in tqdm(loader, desc="Preprocessing"):
        if sample["error"] is not None:
            print(f"Skipped {sample['path']}: {sample['error']}")
            failed.append({key: sample[key] for key in ("path", "error")})
            continue
        blosc2.asarray(
            sample["encoder_input"].numpy(),
            urlpath=str(output_paths[Path(sample["path"])]),
            mode="w",
        )
    if failed:
        save_json(failed, output_dir / "failed.json")
        print(f"Skipped {len(failed)} of {len(files)} volumes, see {output_dir}/failed.json")
    print(
        f"Saved {len(files) - len(failed)} encoder inputs to {output_dir}, settings in {settings_file}"
    )


if __name__ == "__main__":
    main(parse_args())
