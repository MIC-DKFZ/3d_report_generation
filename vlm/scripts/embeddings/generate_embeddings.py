"""
Generate the vision embeddings of the nnFoundation and RMR papers from CT volumes.

'--embedding' names the features as the paper configs do ('data.vision_input_path' is
'${VLM_EMBEDS}/<DATASET>_<embedding>'):
- nnFoundation: nnFoundationViT and nnFoundationCNN, from the ROI. On CT-RATE, nnFoundationViT
  center crops the ROI and nnFoundationCNN resizes it; on Merlin, both resize it.
- RMR: Primus-1B_ROI and Primus-1B_Whole, encoded by PrimusM-MAE as one input block, from the
  ROI or the whole volume; both resize their input.

'--dataset' selects the dataset's layout:
- ctrate: split folders 'train_fixed' and 'valid_fixed'; reoriented to LPS.
- merlin: no split folders; reoriented to LAS.

The ROI embeddings crop CT-RATE to the lungs and Merlin to the foreground; '--roi' replaces it.
Both ROIs come from TotalSegmentator 'total' masks:
- lungs: labels 10-14.
- foreground: all labels, without components of fewer than 400 voxels. Before fitting, the ROI
  is enlarged to at least the encoder input (192 mm at 1 mm) where the volume allows it.

Each volume '<name>.nii.gz' found (recursively) in the split folders of '--input_dir' is
reoriented, z-score normalized, resampled to 1 mm, cropped to its ROI (the mask's bounding box,
enlarged by 4%, from the mask with the same relative path in '--mask_dir'), fitted to the encoder
input (192³) and encoded in float32, on a GPU if one is available: a center crop or pad keeps
1 mm, while resizing gives a spacing of the input's size divided by 192. Volumes without a mask
are embedded whole, with a warning. The embedding
(C, D, H, W) is saved as '<output_dir>/<split folder>/<name>_embedding.bl2', where the paper
configs expect it (see documentation/data_preparation.md). Existing embeddings are skipped, so an
interrupted run can be restarted.

Options beyond the papers: '--processing_mode' replaces the embedding's mode, e.g.
'resized_direct' (the ROI cropped at the volume's own spacing and resized once, with
anti-aliasing; see vlm/data/preprocessing/encoder_input.py), and '--roi' replaces the
paper's ROI, e.g. '--roi foreground' for CT-RATE. Launched with torchrun, the volumes are split
across the processes (on all nodes), each on its own GPU.

The settings are saved once per embedding folder, as '<output_dir>/vision_input_preprocessing.yml';
training copies them into the run config, so that 'inference_nifti.py' embeds new volumes the same
way.

    uv run python vlm/scripts/embeddings/generate_embeddings.py \\
        --embedding nnFoundationViT --dataset ctrate --input_dir /path/to/CT-RATE/dataset \\
        --mask_dir /path/to/CT-RATE/dataset/ts_seg/ts_total \\
        --output_dir "${VLM_EMBEDS}/CTRATE_nnFoundationViT"
"""

import argparse
import os
from pathlib import Path

import blosc2
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm.configs.data_config import VisionInputPreprocessingConfig, VisionInputSettings
from vlm.data.preprocessing.encoder_input import PROCESSING_MODES
from vlm.data.preprocessing.nifti_inputs import (
    NiftiEncoderInputDataset,
    collate_samples,
    find_nifti_files,
    strip_nifti_suffix,
)
from vlm.data.vision_input_settings import save_settings
from vlm.models.vision_encoders.primus import PrimusEncoder
from vlm.models.vision_encoders.resenc import ResEncEncoder
from vlm.utils.serialization import save_json

# Encoder, its default checkpoint, and per dataset its ROI (None: the whole volume) and
# processing mode.
EMBEDDINGS = {
    "nnFoundationViT": {
        "encoder": PrimusEncoder,
        "checkpoint": "MIC-DKFZ/nnFoundationViT",
        "rois": {"ctrate": "lungs", "merlin": "foreground"},
        "processing_modes": {"ctrate": "center_crop", "merlin": "resized"},
    },
    "nnFoundationCNN": {
        "encoder": ResEncEncoder,
        "checkpoint": "MIC-DKFZ/nnFoundationCNN",
        "rois": {"ctrate": "lungs", "merlin": "foreground"},
        "processing_modes": {"ctrate": "resized", "merlin": "resized"},
    },
    "Primus-1B_ROI": {
        "encoder": PrimusEncoder,
        "checkpoint": "MIC-DKFZ/PrimusM-MAE",
        "rois": {"ctrate": "lungs", "merlin": "foreground"},
        "processing_modes": {"ctrate": "resized", "merlin": "resized"},
    },
    "Primus-1B_Whole": {
        "encoder": PrimusEncoder,
        "checkpoint": "MIC-DKFZ/PrimusM-MAE",
        "rois": None,
        "processing_modes": {"ctrate": "resized", "merlin": "resized"},
    },
}

# Default target spacing in millimetres (z, y, x).
TARGET_SPACING = (1.0, 1.0, 1.0)

# ROIs from TotalSegmentator 'total' masks, as recorded in 'data.vision_input_preprocessing'.
ROIS = {
    "lungs": {
        "roi_mask": "TotalSegmentator 'total', lungs (labels 10-14)",
        "mask_label_ids": [10, 11, 12, 13, 14],
        "enforce_min_bbox": False,
        "outlier_method": "none",
    },
    "foreground": {
        "roi_mask": "TotalSegmentator 'total', foreground of all labels",
        "mask_label_ids": None,
        "enforce_min_bbox": True,
        "outlier_method": "connected_components",
        "outlier_min_voxels": 400,
    },
}

# Orientations are those the datasets are mostly stored in, so that most volumes stay unchanged.
DATASETS = {
    "ctrate": {
        "orientation": "LPS",
        "split_folders": ["train_fixed", "valid_fixed"],
    },
    "merlin": {
        "orientation": "LAS",
        "split_folders": [""],
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--embedding", choices=EMBEDDINGS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="The dataset's NIfTI volumes: the folder holding 'train_fixed' and 'valid_fixed' "
        "for CT-RATE (e.g. CT-RATE/dataset), the volumes themselves for Merlin.",
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=None,
        help="TotalSegmentator 'total' masks with the same relative paths as the volumes in "
        "'--input_dir', for the ROI embeddings. Volumes without a mask are embedded whole, with "
        "a warning.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Embedding folder, the 'data.vision_input_path' of the runs; holds the settings.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Encoder checkpoint file or Hugging Face repository; by default the encoder's "
        "repository, e.g. 'MIC-DKFZ/nnFoundationViT'.",
    )
    parser.add_argument(
        "--processing_mode",
        choices=PROCESSING_MODES,
        default=None,
        help="Replaces the embedding's processing mode, e.g. 'resized_direct'.",
    )
    parser.add_argument(
        "--roi",
        choices=ROIS,
        default=None,
        help="Replaces the embedding's ROI (CT-RATE: lungs, Merlin: foreground).",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--overwrite", action="store_true", help="Embed volumes already saved again."
    )
    args = parser.parse_args(argv)
    uses_roi = EMBEDDINGS[args.embedding]["rois"] is not None
    if uses_roi and args.mask_dir is None:
        parser.error(f"{args.embedding} embeds the ROI and needs '--mask_dir'.")
    if not uses_roi and args.mask_dir is not None:
        parser.error(f"{args.embedding} embeds the whole volume and takes no '--mask_dir'.")
    if not uses_roi and args.roi is not None:
        parser.error(f"{args.embedding} embeds the whole volume and takes no '--roi'.")
    return args


def paper_settings(
    embedding: str,
    dataset: str,
    checkpoint: str,
    processing_mode: str | None = None,
    roi: str | None = None,
) -> VisionInputSettings:
    preset = EMBEDDINGS[embedding]
    return VisionInputSettings(
        vision_input_orientation=DATASETS[dataset]["orientation"],
        vision_input_preprocessing=VisionInputPreprocessingConfig(
            target_spacing=TARGET_SPACING,
            normalization="z_score",
            processing_mode=processing_mode or preset["processing_modes"][dataset],
            **(ROIS[roi or preset["rois"][dataset]] if preset["rois"] else {}),
        ),
        embedding_encoder={
            "class_name": preset["encoder"].__name__,
            "checkpoint_path": checkpoint,
        },
    )


def save_embedding(embedding: np.ndarray, path: Path):
    """Save atomically, so that an interrupted run leaves no partial embedding behind."""
    tmp_path = f"{path}.tmp"
    blosc2.save_array(
        np.ascontiguousarray(embedding),
        tmp_path,
        mode="w",
        cparams={"codec": blosc2.Codec.ZSTD, "clevel": 8},
    )
    os.replace(tmp_path, path)


def split_files(
    input_dir: Path, output_dir: Path, overwrite: bool, rank: int = 0, world_size: int = 1
) -> dict[Path, Path]:
    """
    The volumes of a split folder still to embed, mapped to their embedding files: of all its
    volumes, every 'world_size'-th from 'rank' on, so that the processes split them the same way
    however many each has already saved.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Split folder {input_dir} not found.")
    files = find_nifti_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")
    output_paths = {
        file: output_dir / f"{Path(strip_nifti_suffix(file)).name}_embedding.bl2" for file in files
    }
    duplicates = len(files) - len(set(output_paths.values()))
    if duplicates:
        raise ValueError(f"{duplicates} volumes in {input_dir} share a file name with another.")
    shard = list(output_paths.items())[rank::world_size]
    return {file: path for file, path in shard if overwrite or not path.exists()}


def init_preprocessing_worker(worker_id: int):
    # DataLoader already limits PyTorch threads; limit SimpleITK's separate pool too.
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def main(args: argparse.Namespace):
    preset = EMBEDDINGS[args.embedding]
    checkpoint = args.checkpoint or preset["checkpoint"]
    settings = paper_settings(
        args.embedding, args.dataset, checkpoint, args.processing_mode, args.roi
    )
    # Under torchrun, each process embeds every 'world_size'-th volume of each split.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    input_dir = args.input_dir.resolve()
    # All splits are listed first, so that a missing one fails before hours of embedding.
    splits = {
        split: split_files(
            input_dir / split, args.output_dir / split, args.overwrite, rank, world_size
        )
        for split in DATASETS[args.dataset]["split_folders"]
    }
    # Before embedding, so that settings differing from the folder's fail early.
    settings_file = save_settings(args.output_dir, settings)

    device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"
    encoder = preset["encoder"].from_pretrained(checkpoint).to(device).eval()
    preprocessing_kwargs = settings.vision_input_preprocessing.preprocessing_kwargs(
        encoder.patch_size
    )
    for split, output_paths in splits.items():
        output_dir = args.output_dir / split
        output_dir.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(
            NiftiEncoderInputDataset(
                input_dir / split,
                list(output_paths),
                settings.vision_input_orientation,
                None if args.mask_dir is None else args.mask_dir / split,
                preprocessing_kwargs,
            ),
            batch_size=1,
            num_workers=args.num_workers,
            collate_fn=collate_samples,
            worker_init_fn=init_preprocessing_worker,
        )
        failed = []
        with torch.inference_mode():
            for (sample,) in tqdm(
                loader, desc=f"Embedding {split or args.dataset}", disable=rank != 0
            ):
                if sample["error"] is not None:
                    print(f"Skipped {sample['path']}: {sample['error']}")
                    failed.append({key: sample[key] for key in ("path", "error")})
                    continue
                # Float32, as for the papers' embeddings: bf16 changes them noticeably.
                embedding = encoder(sample["encoder_input"][None].to(device).float())[0]
                save_embedding(embedding.cpu().numpy(), output_paths[Path(sample["path"])])
        if failed:
            failed_file = output_dir / (
                "failed.json" if world_size == 1 else f"failed_rank{rank}.json"
            )
            save_json(failed, failed_file)
            print(f"Skipped {len(failed)} of {len(output_paths)} volumes, see {failed_file}")
        print(f"Saved {len(output_paths) - len(failed)} embeddings to {output_dir}")
    print(f"Settings in {settings_file}")


if __name__ == "__main__":
    main(parse_args())
