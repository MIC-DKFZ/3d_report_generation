"""
Generate the vision embeddings of the nnFoundation and RMR papers from CT volumes.

'--embedding' names the features as the paper configs do ('data.vision_input_path' is
'${VLM_EMBEDS}/<DATASET>_<embedding>'):
- nnFoundation: nnFoundationViT and nnFoundationCNN, from the ROI. On CT-RATE, nnFoundationViT
  center crops the ROI and nnFoundationCNN resizes it; on Merlin, both resize it.
- RMR: Primus-1B_ROI and Primus-1B_Whole, encoded by PrimusM-MAE as one input block, from the
  ROI or the whole volume, and Primus-4B_2_1_2_ROI (CT-RATE) and Primus-4B_1_2_2_ROI (Merlin), as
  2 × 1 × 2 or 1 × 2 × 2 input blocks from the ROI; all resize their input.
- nnFoundationViT_release: the released nnFoundationViT, from the foreground, 'resized_direct'.

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
are embedded whole, with a warning. The embedding (C, D, H, W) is saved as
'<output_dir>/<split folder>/<name>_embedding.bl2', where the paper configs expect it (see
documentation/data_preparation.md). Existing embeddings are skipped, so an interrupted run can be
restarted.

The multi-block embeddings (Primus-4B) cover the resized ROI with a window of input blocks,
encoded one by one and merged into one embedding grid (see vlm/embeddings/multiblock.py).

Options beyond the papers ('--help' lists them) change the preset, e.g. '--processing_mode
resized_direct' (the ROI cropped at the volume's own spacing and resized once, with
anti-aliasing; see vlm/data/preprocessing/encoder_input.py) or '--roi foreground' for CT-RATE.
'--orientation' replaces the input orientation (by default the encoder's, if it has one,
otherwise the dataset's).
With '--output_base' instead of '--output_dir', the folder is named after the dataset, the
embedding and these options, e.g. 'CTRATE_nnFoundationViT_roi-foreground'. Launched with
torchrun, the volumes are split across the processes (on all nodes), each on its own GPU;
'--num_shards' and '--shard_id' split them across independent jobs.

The settings of single-block embeddings are saved once per embedding folder, as
'<output_dir>/vision_input_preprocessing.yml'; training copies them into the run config, so that
'inference_nifti.py' embeds new volumes the same way.

    uv run python vlm/scripts/embeddings/generate_embeddings.py \\
        --embedding nnFoundationViT --dataset ctrate --input_dir /path/to/CT-RATE/dataset \\
        --mask_dir /path/to/CT-RATE/dataset/ts_seg/ts_total \\
        --output_dir "${VLM_EMBEDS}/CTRATE_nnFoundationViT"
"""

import argparse
from dataclasses import fields
from pathlib import Path

from vlm.configs.data_config import (
    VisionInputPreprocessingConfig,
    VisionInputSettings,
    validate_orientation,
)
from vlm.data.preprocessing.encoder_input import NORMALIZATIONS, PROCESSING_MODES
from vlm.embeddings.registry import DATASETS, EMBEDDINGS, ENCODERS, ROIS
from vlm.embeddings.runner import EmbeddingJob, RuntimeOptions, run


def orientation_code(value: str) -> str:
    try:
        return validate_orientation(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def default_orientation(args: argparse.Namespace) -> str:
    encoder = ENCODERS[EMBEDDINGS[args.embedding].encoder]
    return encoder.orientation or DATASETS[args.dataset]["orientation"]


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output_dir",
        type=Path,
        help="Embedding folder, the 'data.vision_input_path' of the runs; holds the settings.",
    )
    output.add_argument(
        "--output_base",
        type=Path,
        help="Folder in which the embedding folder is named after the dataset, the embedding "
        "and the options changing it, e.g. 'CTRATE_nnFoundationViT_roi-foreground'.",
    )

    preset = parser.add_argument_group("changes to the embedding preset")
    preset.add_argument(
        "--checkpoint",
        default=None,
        help="Encoder checkpoint file or Hugging Face repository (optionally "
        "'<repository>@<revision>'); by default the embedding's. Replacing the embedding's "
        "checkpoint needs '--output_dir'.",
    )
    preset.add_argument("--roi", choices=ROIS, default=None, help="Replaces the embedding's ROI.")
    preset.add_argument(
        "--processing_mode",
        choices=PROCESSING_MODES,
        default=None,
        help="Replaces the embedding's processing mode, e.g. 'resized_direct' (the ROI cropped "
        "at the volume's own spacing and resized once, with anti-aliasing).",
    )
    preset.add_argument(
        "--grid",
        type=int,
        nargs=3,
        default=None,
        metavar=("GZ", "GY", "GX"),
        help="Encoder inputs per axis: a window of GZ·GY·GX inputs is embedded block-wise and "
        "the block embeddings merged (no settings file is saved).",
    )
    preset.add_argument(
        "--tile_step_size",
        type=float,
        default=1.0,
        help="Multi-block: sliding-window step as a fraction of the input; below 1, blocks "
        "overlap and are merged with Gaussian weights.",
    )
    preset.add_argument("--input_shape", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    preset.add_argument(
        "--target_spacing", type=float, nargs=3, default=None, metavar=("Z", "Y", "X")
    )
    preset.add_argument("--normalization", choices=NORMALIZATIONS, default=None)
    preset.add_argument(
        "--orientation",
        type=orientation_code,
        default=None,
        help="Orientation of the encoder inputs, e.g. 'RAS'; by default the encoder's, otherwise "
        "the dataset's (CT-RATE 'LPS', Merlin 'LAS').",
    )
    preset.add_argument(
        "--bf16",
        action="store_true",
        help="Encode in bfloat16 (saved as float32); changes the embeddings noticeably, but "
        "large multi-block inputs need it for memory-efficient attention.",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--cudnn_attention",
        action="store_true",
        help="Attention through cuDNN with the head dim zero-padded (needs '--bf16'); faster "
        "on GH200, differing by rounding only.",
    )
    runtime.add_argument("--num_workers", type=int, default=4)
    runtime.add_argument("--prefetch_factor", type=int, default=2)
    runtime.add_argument(
        "--batch_size", type=int, default=1, help="Single-block volumes per batch."
    )
    runtime.add_argument(
        "--block_batch_size", type=int, default=1, help="Multi-block: blocks per encoder call."
    )
    runtime.add_argument(
        "--save_workers",
        type=int,
        default=2,
        help="Processes saving the embeddings; 0 saves them in the main process.",
    )
    runtime.add_argument("--save_threads", type=int, default=8, help="blosc2 threads per saver.")
    runtime.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Independent jobs the volumes are split across, e.g. one per node.",
    )
    runtime.add_argument("--shard_id", type=int, default=0, help="This job's shard.")
    runtime.add_argument("--max_files", type=int, default=None, help="Per split, for testing.")
    runtime.add_argument(
        "--debug_save_preprocessed",
        action="store_true",
        help="Also save the encoder inputs as NIfTI in '<split folder>/preprocessed'.",
    )
    runtime.add_argument(
        "--overwrite", action="store_true", help="Embed volumes already saved again."
    )
    return parser


def validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    preset = EMBEDDINGS[args.embedding]
    uses_roi = preset.rois is not None
    if uses_roi and args.mask_dir is None:
        parser.error(f"{args.embedding} embeds the ROI and needs '--mask_dir'.")
    if not uses_roi and args.mask_dir is not None:
        parser.error(f"{args.embedding} embeds the whole volume and takes no '--mask_dir'.")
    if not uses_roi and args.roi is not None:
        parser.error(f"{args.embedding} embeds the whole volume and takes no '--roi'.")
    if preset.checkpoint is None and args.checkpoint is None:
        parser.error(f"{args.embedding} has no checkpoint of its own and needs '--checkpoint'.")
    if preset.checkpoint is not None and args.checkpoint is not None and args.output_dir is None:
        parser.error("Another '--checkpoint' needs an '--output_dir'.")
    grid = tuple(args.grid or preset.grid)
    if grid == (1, 1, 1) and args.tile_step_size != 1.0:
        parser.error("'--tile_step_size' needs a multi-block grid.")
    if not 0 < args.tile_step_size <= 1:
        parser.error("'--tile_step_size' must be in (0, 1].")
    if args.cudnn_attention and not args.bf16:
        parser.error("'--cudnn_attention' needs '--bf16'.")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("'--shard_id' must be in [0, '--num_shards').")
    return args


def parse_args(argv: list[str] | None = None, description: str | None = None):
    parser = build_parser(description)
    return validate(parser, parser.parse_args(argv))


def folder_name(args: argparse.Namespace) -> str:
    """'<DATASET>_<embedding>', followed by the options that change the embeddings."""
    preset = EMBEDDINGS[args.embedding]
    parts = [args.dataset.upper(), args.embedding]
    if args.roi and args.roi != preset.roi(args.dataset):
        parts.append(f"roi-{args.roi}")
    if args.processing_mode and args.processing_mode != preset.processing_mode(args.dataset):
        parts.append(args.processing_mode)
    if args.grid and tuple(args.grid) != preset.grid:
        parts.append("grid" + "x".join(map(str, args.grid)))
    if args.tile_step_size != 1.0:
        parts.append(f"step{args.tile_step_size:g}")
    if args.input_shape:
        parts.append("in" + "x".join(map(str, args.input_shape)))
    if args.target_spacing:
        parts.append("sp" + "x".join(f"{s:g}" for s in args.target_spacing))
    if args.normalization:
        parts.append(args.normalization)
    if args.orientation and args.orientation != default_orientation(args):
        parts.append(args.orientation)
    if args.bf16:
        parts.append("bf16")
    return "_".join(parts)


def embedding_job(args: argparse.Namespace) -> EmbeddingJob:
    preset = EMBEDDINGS[args.embedding]
    encoder = ENCODERS[preset.encoder]
    checkpoint = args.checkpoint or preset.checkpoint
    roi = args.roi or preset.roi(args.dataset)
    settings = VisionInputSettings(
        vision_input_orientation=args.orientation or default_orientation(args),
        vision_input_preprocessing=VisionInputPreprocessingConfig(
            target_spacing=args.target_spacing or encoder.target_spacing,
            normalization=args.normalization or preset.normalization or encoder.normalization,
            processing_mode=args.processing_mode or preset.processing_mode(args.dataset),
            **(ROIS[roi] if roi else {}),
        ),
        embedding_encoder=(
            None
            if encoder.embedding_encoder is None or checkpoint is None
            else {"class_name": encoder.embedding_encoder, "checkpoint_path": checkpoint}
        ),
    )
    return EmbeddingJob(
        build_encoder=lambda: encoder.build(checkpoint),
        settings=settings,
        input_dir=args.input_dir,
        output_dir=args.output_dir or args.output_base / folder_name(args),
        split_folders=DATASETS[args.dataset]["split_folders"],
        mask_dir=args.mask_dir if roi else None,
        input_shape=args.input_shape or encoder.input_shape,
        grid=tuple(args.grid or preset.grid),
        tile_step_size=args.tile_step_size,
        bf16=args.bf16,
        options=RuntimeOptions(**{f.name: getattr(args, f.name) for f in fields(RuntimeOptions)}),
    )


def main(argv: list[str] | None = None, description: str | None = None) -> None:
    """Runs the command line; packages with their own presets register them and call it."""
    run(embedding_job(parse_args(argv, description or __doc__.split("\n\n")[0].strip())))


if __name__ == "__main__":
    main()
