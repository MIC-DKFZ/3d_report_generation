"""
Generate reports for a folder of NIfTI volumes with a trained run.

Unlike 'inference.py', which runs on the dataset of the run's config, this script takes any
folder of (nested) NIfTI files. Each volume is reoriented to 'data.vision_input_orientation',
preprocessed like the volumes of the training embeddings, encoded by the frozen vision encoder and
passed to the trained model, so the model receives the same kind of embedding as in training.

The encoder and the dataset-specific preprocessing (ROI masks, processing mode) are not part of
the run's config, since the run was trained on precomputed embeddings; pass them as arguments.
Dotted config overrides are accepted as in 'inference.py', e.g.
'--data.vision_input_orientation LPS' for runs whose config does not set it.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm.configs.model.vision_encoder_config import VisionEncoderClassConfig
from vlm.data.batches import extract_batch_prompt_only
from vlm.data.datasets.base import QUESTION, SYSTEM_TEXT, format_into_message
from vlm.data.preprocessing.encoder_input import PROCESSING_MODES
from vlm.data.preprocessing.nifti_inputs import (
    NiftiEncoderInputDataset,
    collate_samples,
    find_nifti_files,
)
from vlm.models.abstract_vl_wrapper import AbstractVLWrapper
from vlm.training.checkpointing import save_final_config
from vlm.utils import logger
from vlm.utils.decoding import decode_generated_answers, strip_thinking
from vlm.utils.logging import log_banner, setup_logger
from vlm.utils.paths import resolve_run_dir
from vlm.utils.run_naming import add_datetime_tag
from vlm.utils.seeding import seed_everything
from vlm.utils.serialization import save_json


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

    encoder = parser.add_argument_group("vision encoder of the training embeddings")
    encoder.add_argument(
        "--vision_encoder",
        default="PrimusEncoder",
        help="Class name in 'vlm/models/vision_encoders': 'PrimusEncoder' (nnFoundationViT) or "
        "'ResEncEncoder' (nnFoundationCNN).",
    )
    encoder.add_argument(
        "--vision_encoder_checkpoint",
        required=True,
        help="Checkpoint file or Hugging Face repository, e.g. 'MIC-DKFZ/nnFoundationViT' or "
        "'MIC-DKFZ/nnFoundationCNN'.",
    )

    preprocessing = parser.add_argument_group(
        "preprocessing, as used for the training embeddings (see embedd_images_baselines.py)"
    )
    preprocessing.add_argument(
        "--processing_mode",
        choices=PROCESSING_MODES,
        required=True,
        help="Resize the ROI to the encoder input shape, or center crop/pad it.",
    )
    preprocessing.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help="Resample to this spacing in mm, for embeddings made with resampling. Defaults to "
        "the spacing of the encoder's pre-training (none for nnFoundationViT).",
    )
    preprocessing.add_argument(
        "--mask_dir",
        type=Path,
        default=None,
        help="ROI masks with the same relative paths as the volumes. Without, the whole volume.",
    )
    preprocessing.add_argument(
        "--mask_label_ids",
        type=int,
        nargs="+",
        default=None,
        help="Mask labels of the ROI; all non-zero labels by default.",
    )
    preprocessing.add_argument(
        "--no_enforce_min_bbox",
        dest="enforce_min_bbox",
        action="store_false",
        help="Do not enlarge the ROI to at least the encoder input shape.",
    )
    preprocessing.add_argument(
        "--outlier_method", choices=["none", "connected_components"], default="none"
    )
    preprocessing.add_argument("--outlier_min_voxels", type=int, default=400)

    prompt = parser.add_argument_group("prompt; defaults to the one of the dataset JSONs")
    prompt.add_argument("--question", default=QUESTION)
    prompt.add_argument("--system_text", default=SYSTEM_TEXT)

    parser.add_argument("--batch_size", type=int, default=1, help="Volumes per batch.")
    parser.add_argument("--num_workers", type=int, default=2, help="Preprocessing workers.")
    return parser.parse_known_args()


def main(args: argparse.Namespace, config_cli_args: list[str]):
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
    setup_logger(Path(output_dir, add_datetime_tag("inference_nifti") + ".log"))

    orientation = cfg.data.vision_input_orientation
    if orientation is None:
        raise ValueError(
            "The run's config does not set 'data.vision_input_orientation', the orientation of "
            "the training volumes; pass it, e.g. '--data.vision_input_orientation LPS'."
        )
    question, system_text = args.question, args.system_text

    # In float32: the embeddings of nnFoundationViT in bfloat16 differ by ~12% from float32.
    encoder = (
        VisionEncoderClassConfig(
            class_name=args.vision_encoder, checkpoint_path=args.vision_encoder_checkpoint
        )
        .to_instance()
        .cuda()
        .eval()
    )
    preprocessing_kwargs = {
        "target_spacing": tuple(args.target_spacing or ()) or encoder.target_spacing,
        "normalization": encoder.normalization,
        "input_shape": encoder.patch_size,
        "processing_mode": args.processing_mode,
        "mask_label_ids": args.mask_label_ids,
        "enforce_min_bbox": args.enforce_min_bbox,
        "outlier_method": args.outlier_method,
        "outlier_min_voxels": args.outlier_min_voxels,
    }
    files = find_nifti_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")
    logger.info_main(
        f"Generating reports for {len(files)} volumes in {input_dir}: reoriented to "
        f"{orientation}, preprocessed with {preprocessing_kwargs}, encoded by "
        f"{args.vision_encoder}. Prompt: {question!r}"
    )
    loader = DataLoader(
        NiftiEncoderInputDataset(
            input_dir, files, orientation, args.mask_dir, preprocessing_kwargs
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
    )

    model = vlm_model.model.cuda().eval()
    expected_shape = (cfg.data.embed_dim, *cfg.data.embed_size)
    results, failed = [], []
    with torch.inference_mode():
        for samples in tqdm(loader):
            for sample in samples:
                if sample["error"] is not None:
                    logger.error(f"Skipped {sample['path']}: {sample['error']}")
                    failed.append({key: sample[key] for key in ("image_id", "path", "error")})
            samples = [s for s in samples if s["error"] is None]
            if not samples:
                continue

            encoder_inputs = torch.stack([s["encoder_input"] for s in samples])
            embeddings = encoder(encoder_inputs.cuda()).cpu()
            if tuple(embeddings.shape[1:]) != expected_shape:
                raise ValueError(
                    f"{args.vision_encoder} gives embeddings of shape "
                    f"{tuple(embeddings.shape[1:])}, but the run was trained on {expected_shape}."
                )

            examples = []
            for sample, embedding in zip(samples, embeddings):
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
                        "vision_input": embedding.numpy(),
                    }
                )
            inputs = vlm_model.collate_fn(examples)
            del inputs["raw_samples"], inputs["batch_count"]
            inputs = extract_batch_prompt_only(inputs).to("cuda")
            # Use the mixed precision of training, as 'inference.py'.
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
                }
                for sample, report in zip(samples, reports)
            ]
            logger.info(f"{samples[0]['image_id']}:\n{reports[0]}")

    save_json(results, Path(output_dir, "inference_results.json"))
    save_json(
        {
            "checkpoint_dir": str(ckpt_dir),
            "input_dir": str(input_dir),
            "vision_input_orientation": orientation,
            "vision_encoder": args.vision_encoder,
            "vision_encoder_checkpoint": args.vision_encoder_checkpoint,
            "mask_dir": None if args.mask_dir is None else str(args.mask_dir),
            "preprocessing": preprocessing_kwargs,
            "question": question,
            "system_text": system_text,
            "failed": failed,
        },
        Path(output_dir, "inference_nifti_settings.json"),
    )
    save_final_config(cfg, output_dir=output_dir)
    if failed:
        logger.warning(f"Skipped {len(failed)} of {len(files)} volumes, see the settings file.")
    log_banner(f"Saved {len(results)} reports to {output_dir}")


if __name__ == "__main__":
    main(*parse_args())
