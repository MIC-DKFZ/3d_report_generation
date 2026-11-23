"""
Generate reports for a folder of NIfTI volumes with a trained run.

Unlike 'inference.py', which runs on the dataset of the run's config, this script takes any
folder of (nested) NIfTI files. Each volume is reoriented to 'data.vision_input_orientation',
preprocessed like the training inputs, encoded by the vision encoder and passed to the trained
model, so the model receives the same kind of input as in training. Embeddings of several
encoder inputs ('data.vision_input_preprocessing.grid', e.g. Primus-4B) are encoded block by
block and merged, as by 'vlm/scripts/embeddings/generate_embeddings.py'.

The preprocessing comes from 'data.vision_input_preprocessing' of the run's config; arguments
override it, and runs without it need at least '--processing_mode' (spacing and normalization
then default to those of the encoder's pre-training). Runs trained on images ('image' mode)
encode with their own vision encoder, as in training; runs trained on precomputed embeddings
encode with the encoder that made them, 'data.embedding_encoder' of the config (or
'--vision_encoder' and '--vision_encoder_checkpoint'). Training fills these config fields from
the dataset's 'vision_input_preprocessing.yml', if it has one. Dotted config overrides are
accepted as in 'inference.py', e.g. '--data.vision_input_orientation LPS' for runs whose config
does not set it.

Runs whose inputs were cropped to an ROI mask ('data.vision_input_preprocessing.roi_mask') crop
the volumes to the masks in '--mask_dir' (same relative paths as the volumes). If the ROI comes
from TotalSegmentator 'total' masks, as for the ROI embeddings of
'vlm/scripts/embeddings/generate_embeddings.py', masks missing there are made with
TotalSegmentator, as the training masks, and saved there for later runs (see
'vlm/data/preprocessing/totalsegmentator_masks.py'); without '--mask_dir', all masks are made, in
'<output_dir>/totalsegmentator_masks'. Other masks, and the masks of runs whose config does not
record the ROI, are not made here and must be passed. Volumes without a usable mask are skipped
(listed under 'failed'), or used whole with '--allow_missing_masks'.

Launched with torchrun, e.g. 'torchrun --standalone --nproc_per_node=4 inference_nifti.py ...',
the volumes are split across the GPUs and rank 0 saves the reports of all of them.
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm.configs.data_config import (
    EmbeddingEncoderConfig,
    VisionInputPreprocessingConfig,
)
from vlm.configs.model.vision_encoder_config import VisionEncoderClassConfig
from vlm.data.batches import extract_batch_prompt_only
from vlm.data.datasets.base import QUESTION, SYSTEM_TEXT, format_into_message
from vlm.data.preprocessing.encoder_input import (
    NORMALIZATIONS,
    OUTLIER_METHODS,
    PROCESSING_MODES,
)
from vlm.data.preprocessing import totalsegmentator_masks
from vlm.data.preprocessing.nifti_inputs import (
    NiftiEncoderInputDataset,
    collate_samples,
    find_nifti_files,
    strip_nifti_suffix,
)
from vlm.data.types import VisionInputType
from vlm.embeddings.multiblock import encode_multiblock, multiblock_sample
from vlm.models.abstract_vl_wrapper import AbstractVLWrapper
from vlm.models.vision_encoders.base import VisionEncoderBase
from vlm.training.checkpointing import save_final_config
from vlm.utils import logger
from vlm.utils.classes import getattr_nested
from vlm.utils.ddp import gather_inference_outputs, is_main_process, setup_ddp
from vlm.utils.decoding import decode_generated_answers, strip_thinking
from vlm.utils.logging import log_banner, setup_logger
from vlm.utils.paths import resolve_run_dir
from vlm.utils.run_naming import add_datetime_tag
from vlm.utils.precision import (
    inference_precision_context,
    prepare_model_for_inference_precision,
)
from vlm.utils.seeding import seed_everything
from vlm.utils.serialization import save_json

# The fields of 'data.vision_input_preprocessing' besides the ROI mask.
PREPROCESSING_FIELDS = (
    "target_spacing",
    "normalization",
    "processing_mode",
    "mask_label_ids",
    "enforce_min_bbox",
    "outlier_method",
    "outlier_min_voxels",
    "grid",
    "tile_step_size",
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Generate reports for a folder of NIfTI volumes with a trained run.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Folder searched recursively for NIfTI files.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint_dir", type=str, help="Checkpoint directory to load.")
    group.add_argument(
        "--run_dir",
        type=str,
        help="Run directory (or unique run folder name under VLM_OUTPUT_DIR); see '--mode'.",
    )
    parser.add_argument(
        "--mode",
        choices=["best", "final"],
        default="final",
        help="Checkpoint of '--run_dir' to load.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to '<checkpoint_dir>/inference_nifti_<input_dir name>'.",
    )

    encoder = parser.add_argument_group(
        "vision encoder of the training embeddings; overrides 'data.embedding_encoder' of the run's "
        "config (runs trained on images use their own)"
    )
    encoder.add_argument(
        "--vision_encoder",
        default=None,
        help="Class name in 'vlm/models/vision_encoders': 'PrimusEncoder' (nnFoundationViT) or "
        "'ResEncEncoder' (nnFoundationCNN).",
    )
    encoder.add_argument(
        "--vision_encoder_checkpoint",
        default=None,
        help="Checkpoint file or Hugging Face repository, '<repository>@<revision>' for a "
        "revision, e.g. 'MIC-DKFZ/nnFoundationViT' or 'MIC-DKFZ/nnFoundationCNN', e.g. a local "
        "copy of the recorded one.",
    )

    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=None,
        help="ROI masks with the same relative paths as the volumes, for runs whose inputs were "
        "cropped to a mask ('data.vision_input_preprocessing.roi_mask'). For TotalSegmentator "
        "'total' masks, missing ones are made and saved here; defaults to "
        "'<output_dir>/totalsegmentator_masks'.",
    )
    parser.add_argument(
        "--no_generate_masks",
        action="store_true",
        help="Do not make missing masks with TotalSegmentator; volumes without a mask are "
        "skipped.",
    )
    parser.add_argument(
        "--allow_missing_masks",
        action="store_true",
        help="Use volumes without a mask, or with an empty one (e.g. where TotalSegmentator "
        "failed), whole instead of skipping them. The model saw almost no whole volumes in "
        "training, so their reports are less reliable; they are marked with "
        '\'"mask": "missing, whole volume"\' in the results.',
    )
    preprocessing = parser.add_argument_group(
        "preprocessing; overrides 'data.vision_input_preprocessing' of the run's config"
    )
    preprocessing.add_argument("--processing_mode", choices=PROCESSING_MODES, default=None)
    preprocessing.add_argument(
        "--target_spacing", type=float, nargs=3, default=None, metavar=("Z", "Y", "X")
    )
    preprocessing.add_argument("--normalization", choices=NORMALIZATIONS, default=None)
    preprocessing.add_argument("--mask_label_ids", type=int, nargs="+", default=None)
    preprocessing.add_argument(
        "--no_enforce_min_bbox",
        dest="enforce_min_bbox",
        action="store_const",
        const=False,
        default=None,
        help="Do not enlarge the ROI to at least the encoder input shape.",
    )
    preprocessing.add_argument("--outlier_method", choices=OUTLIER_METHODS, default=None)
    preprocessing.add_argument("--outlier_min_voxels", type=int, default=None)
    preprocessing.add_argument(
        "--grid",
        type=int,
        nargs=3,
        default=None,
        metavar=("GZ", "GY", "GX"),
        help="Encoder inputs per axis of multi-block embeddings (e.g. 2 1 2 for Primus-4B on "
        "CT-RATE).",
    )
    preprocessing.add_argument("--tile_step_size", type=float, default=None)

    prompt = parser.add_argument_group("prompt; defaults to the one of the dataset JSONs")
    prompt.add_argument("--question", default=QUESTION)
    prompt.add_argument("--system_text", default=SYSTEM_TEXT)

    parser.add_argument("--batch_size", type=int, default=1, help="Volumes per batch.")
    parser.add_argument("--num_workers", type=int, default=2, help="Preprocessing workers.")
    return parser.parse_known_args()


def resolve_preprocessing(
    args: argparse.Namespace,
    configured: VisionInputPreprocessingConfig | None,
    encoder: VisionEncoderBase,
) -> tuple[VisionInputPreprocessingConfig, dict[str, str]]:
    """
    The preprocessing and where each setting comes from: the arguments, else the run's config,
    else (for runs without 'data.vision_input_preprocessing') the encoder's pre-training for the
    spacing and normalization and the defaults for the mask settings.
    """
    values, sources = {}, {}
    for name in PREPROCESSING_FIELDS:
        if getattr(args, name) is not None:
            values[name], sources[name] = getattr(args, name), "argument"
        elif configured is not None:
            values[name], sources[name] = getattr(configured, name), "run config"
        elif name in ("target_spacing", "normalization"):
            values[name], sources[name] = getattr(encoder, name), "encoder"
        elif name == "processing_mode":
            raise ValueError(
                "The run's config does not set 'data.vision_input_preprocessing'; pass at least "
                "'--processing_mode', as used for the training inputs."
            )
        else:
            values[name] = VisionInputPreprocessingConfig.model_fields[name].default
            sources[name] = "default"
    if configured is not None:
        roi_mask = configured.roi_mask
    else:
        roi_mask = None if args.mask_dir is None else f"masks in {args.mask_dir}"

    if roi_mask is not None and args.mask_dir is None:
        raise ValueError(
            f"The run's inputs were cropped to an ROI mask ({roi_mask}); pass the masks of the "
            "volumes with '--mask_dir'."
        )
    if configured is not None and roi_mask is None and args.mask_dir is not None:
        logger.warning(
            "The run's inputs were whole volumes, but '--mask_dir' crops the volumes to masks."
        )
    return VisionInputPreprocessingConfig(roi_mask=roi_mask, **values), sources


def multiblock_preprocess(
    preprocessing: VisionInputPreprocessingConfig, preprocessing_kwargs: dict
) -> Callable | None:
    """The 'preprocess' of 'NiftiEncoderInputDataset' for multi-block inputs, else None."""
    if not preprocessing.multiblock:
        return None
    return partial(
        multiblock_sample,
        **preprocessing_kwargs,
        grid=tuple(preprocessing.grid),
        tile_step_size=preprocessing.tile_step_size,
    )


def encode_samples(
    encoder: VisionEncoderBase,
    samples: list[dict],
    preprocessing: VisionInputPreprocessingConfig,
    input_shape: tuple[int, int, int],
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """The embeddings (n, C, D, H, W) of the samples, on the CPU."""
    if not preprocessing.multiblock:
        inputs = torch.stack([s["encoder_input"] for s in samples])
        return encoder(inputs.to(device)).cpu()
    return torch.stack(
        [
            encode_multiblock(
                lambda blocks: encoder(blocks.to(device)),
                sample,
                tuple(input_shape),
                tuple(preprocessing.grid),
            ).cpu()
            for sample in samples
        ]
    )


def resolve_embedding_encoder(
    args: argparse.Namespace, recorded: EmbeddingEncoderConfig | None
) -> tuple[str, str]:
    """The class and checkpoint of the encoder that made the embeddings; arguments override the
    run config's record."""
    class_name = args.vision_encoder or (recorded and recorded.class_name)
    checkpoint = args.vision_encoder_checkpoint or (recorded and recorded.checkpoint_path)
    if not (class_name and checkpoint):
        raise ValueError(
            "The run's config does not record the encoder that made its embeddings "
            "('data.embedding_encoder'); pass it with '--vision_encoder' and "
            "'--vision_encoder_checkpoint'."
        )
    return class_name, checkpoint


def main(args: argparse.Namespace, config_cli_args: list[str]):
    is_distributed, rank, world_size, _, _ = setup_ddp()
    seed_everything(0)
    if args.run_dir:
        vlm_model, cfg, ckpt_dir = AbstractVLWrapper.load_from_run_dir(
            resolve_run_dir(args.run_dir), args.mode
        )
    else:
        ckpt_dir = Path(args.checkpoint_dir)
        vlm_model, cfg = AbstractVLWrapper.load_from_checkpoint_dir(ckpt_dir)
    cfg.merge_subconfigs_with_cli_args(config_cli_args)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir or Path(ckpt_dir, f"inference_nifti_{input_dir.name}")
    # All ranks log to the file named with rank 0's timestamp, as in 'inference.py'.
    log_file_names = [add_datetime_tag("inference_nifti") + ".log" if is_main_process() else None]
    if is_distributed:
        dist.broadcast_object_list(log_file_names, src=0)
    setup_logger(Path(output_dir, log_file_names[0]))

    orientation = cfg.data.vision_input_orientation
    if orientation is None:
        raise ValueError(
            "The run's config does not set 'data.vision_input_orientation', the orientation of "
            "the training volumes; pass it, e.g. '--data.vision_input_orientation LPS'."
        )
    question, system_text = args.question, args.system_text

    image_mode = cfg.data.vision_input_type == VisionInputType.IMAGE
    if image_mode:
        # Encoded inside the model by the run's (possibly fine-tuned) encoder, as in training.
        if args.vision_encoder or args.vision_encoder_checkpoint:
            raise ValueError(
                "The run was trained on images and encodes them with its own vision encoder; "
                "do not pass '--vision_encoder' or '--vision_encoder_checkpoint'."
            )
        encoder = getattr_nested(vlm_model.model, vlm_model.module_attr_map["vision_encoder"])
        encoder_name = f"the run's {cfg.model.vision_encoder.class_name}"
        checkpoint = None
        input_shape = cfg.model.vision_encoder.patch_size or encoder.patch_size
    else:
        encoder_name, checkpoint = resolve_embedding_encoder(args, cfg.data.embedding_encoder)
        # In float32: the embeddings of nnFoundationViT in bfloat16 differ by ~12% from float32.
        encoder = (
            VisionEncoderClassConfig(class_name=encoder_name, checkpoint_path=checkpoint)
            .to_instance()
            .cuda()
            .eval()
        )
        input_shape = encoder.patch_size
    configured = cfg.data.vision_input_preprocessing
    generate_masks = (
        not args.no_generate_masks
        and configured is not None
        and totalsegmentator_masks.can_generate(configured.roi_mask)
    )
    if generate_masks and args.mask_dir is None:
        args.mask_dir = Path(output_dir, "totalsegmentator_masks")
    preprocessing, sources = resolve_preprocessing(args, configured, encoder)
    if image_mode and preprocessing.multiblock:
        raise ValueError("Runs trained on images take one encoder input; 'grid' must be 1 1 1.")
    preprocessing_kwargs = preprocessing.preprocessing_kwargs(input_shape)
    preprocess = multiblock_preprocess(preprocessing, preprocessing_kwargs)
    described = {
        **preprocessing_kwargs,
        "grid": tuple(preprocessing.grid),
        "tile_step_size": preprocessing.tile_step_size,
    }
    files = find_nifti_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")
    logger.info_main(
        f"Generating reports for {len(files)} volumes in {input_dir} on {world_size} GPU(s): "
        f"reoriented to {orientation}, preprocessed with {described} (from: "
        f"{sources}), encoded by {encoder_name}. Prompt: {question!r}"
    )
    # Each rank takes every 'world_size'-th volume.
    rank_files = files[rank::world_size]
    mask_sources = {}
    if generate_masks:
        mask_sources = totalsegmentator_masks.generate_missing_masks(
            input_dir, rank_files, args.mask_dir, device=f"gpu:{torch.cuda.current_device()}"
        )
        # Free TotalSegmentator's GPU memory for the model.
        torch.cuda.empty_cache()
    elif args.mask_dir is not None:
        mask_sources = {
            strip_nifti_suffix(path.relative_to(input_dir)): "read" for path in rank_files
        }
    loader = DataLoader(
        NiftiEncoderInputDataset(
            input_dir,
            rank_files,
            orientation,
            args.mask_dir,
            preprocessing_kwargs,
            strict_masks=True,
            allow_missing_masks=args.allow_missing_masks,
            preprocess=preprocess,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
    )

    model = vlm_model.model.cuda().eval()
    precision = cfg.eval.inference_precision
    prepare_model_for_inference_precision(model, precision)
    expected_shape = None if image_mode else (cfg.data.embed_dim, *cfg.data.embed_size)
    results, failed = [], []
    with torch.inference_mode():
        for samples in tqdm(loader, disable=not is_main_process()):
            for sample in samples:
                if sample["error"] is not None:
                    logger.error(f"Skipped {sample['path']}: {sample['error']}")
                    failed.append({key: sample[key] for key in ("image_id", "path", "error")})
            samples = [s for s in samples if s["error"] is None]
            if not samples:
                continue

            if image_mode:
                vision_inputs = [s["encoder_input"] for s in samples]
            else:
                vision_inputs = encode_samples(encoder, samples, preprocessing, input_shape)
                if tuple(vision_inputs.shape[1:]) != expected_shape:
                    raise ValueError(
                        f"{encoder_name} gives embeddings of shape "
                        f"{tuple(vision_inputs.shape[1:])}, but the run was trained on "
                        f"{expected_shape}."
                    )

            examples = []
            for sample, vision_input in zip(samples, vision_inputs):
                formatted_qa, formatted_qa_prompt_only = format_into_message(
                    question, "", system_text
                )
                examples.append(
                    {
                        "image_id": sample["image_id"],
                        "question": question,
                        "full_answer": "",
                        "answer_type": None,
                        "answer_subtype": None,
                        "formatted_qa": formatted_qa,
                        "formatted_qa_prompt_only": formatted_qa_prompt_only,
                        "vision_input": vision_input.numpy(),
                    }
                )
            inputs = vlm_model.collate_fn(examples)
            del inputs["raw_samples"], inputs["batch_count"]
            inputs = extract_batch_prompt_only(inputs).to("cuda")
            with inference_precision_context(precision):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=cfg.eval.max_new_tokens,
                    **cfg.eval.generation_kwargs(),
                )
            reports = decode_generated_answers(vlm_model.tokenizer, outputs, inputs["input_ids"])
            results += [
                {
                    "image_id": sample["image_id"],
                    "path": sample["path"],
                    "output_text": report,
                    "output_text_clean": strip_thinking(report),
                    "mask": (
                        "missing, whole volume"
                        if sample["whole_volume"]
                        else mask_sources.get(sample["image_id"])
                    ),
                }
                for sample, report in zip(samples, reports)
            ]
            logger.info(f"{samples[0]['image_id']}:\n{reports[0]}")

    if is_distributed:
        # Generation can leave the GPU almost full in cached blocks, and the gather needs
        # GPU memory: NCCL then fails with "unhandled cuda error".
        torch.cuda.empty_cache()
        gathered = gather_inference_outputs({"results": results, "failed": failed}, main_rank=0)
        if is_main_process():
            results, failed = gathered["results"], gathered["failed"]
    if is_main_process():
        results.sort(key=lambda r: r["image_id"])
        failed.sort(key=lambda f: f["image_id"])
        save_json(results, Path(output_dir, "inference_results.json"))
        save_json(
            {
                "checkpoint_dir": str(ckpt_dir),
                "input_dir": str(input_dir),
                "vision_input_orientation": orientation,
                "vision_encoder": encoder_name,
                "vision_encoder_checkpoint": checkpoint,
                "mask_dir": None if args.mask_dir is None else str(args.mask_dir),
                "generated_masks": sum(r["mask"] == "generated" for r in results),
                "allow_missing_masks": args.allow_missing_masks,
                "whole_volumes": sum(r["mask"] == "missing, whole volume" for r in results),
                "totalsegmentator": totalsegmentator_masks.settings() if generate_masks else None,
                "roi_mask": preprocessing.roi_mask,
                "preprocessing": described,
                "preprocessing_sources": sources,
                "inference_precision": precision,
                "question": question,
                "system_text": system_text,
                "failed": failed,
            },
            Path(output_dir, "inference_nifti_settings.json"),
        )
        save_final_config(cfg, output_dir=output_dir)
        if failed:
            logger.warning(
                f"Skipped {len(failed)} of {len(files)} volumes, see the settings file."
            )
        log_banner(f"Saved {len(results)} reports to {output_dir}")
    # The other ranks wait until rank 0 has saved the outputs.
    if is_distributed:
        dist.barrier()


if __name__ == "__main__":
    main(*parse_args())
