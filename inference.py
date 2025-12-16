import argparse
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from vlm.configs.run_config import RunConfig
from vlm.data.datasets import AMOSMMDataset, CTRATEDataset
from vlm.evaluation.eval_distributed_sampler import EvalDistributedSampler
from vlm.models.abstract_vlm import AbstractVLMModel
from vlm.utils import logger
from vlm.utils.ddp_utils import setup_ddp, gather_inference_outputs
from vlm.utils.filter_batch import extract_batch_prompt_only
from vlm.utils.helper import is_main_process, find_run_dir_in_output_dir, seed_everything, merge_config_with_cli_args, \
    add_datetime_tag
from vlm.utils.load_and_save_files import load_json, save_json
from vlm.utils.logging import setup_logger, log_banner
from eval import main as eval_main
from vlm.utils.vision_input_type import VisionInputType


class UnifiedVLMInference:
    def __init__(self, vlm_model: AbstractVLMModel, cfg: RunConfig, output_dir: str | Path, per_device_batch_size: int,
                 num_dataloader_workers: int, max_new_tokens: int):
        """
        Initialize the unified VLM inference.
        Args:
            vlm_model: An instance of a class derived from AbstractVLMModel
            cfg: The RunConfig instance from the training run
            output_dir: Directory to save inference outputs
        """
        self.vlm_model = vlm_model
        self.cfg = cfg
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)

        self.per_device_batch_size = per_device_batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.max_new_tokens = max_new_tokens

        (self.is_distributed,
         self.rank,
         self.world_size,
         self.local_rank,
         self.local_world_size) = setup_ddp()

    def inference(self, test_dataset):
        """
        Main inference function that runs inference on the VLM model.
        Args:
            test_dataset: Test dataset
        """
        model = self.vlm_model.model
        model.bfloat16().cuda().eval()

        if self.is_distributed:
            # Wrap the model with DDP
            logger.info(f"Wrapping model with DDP on rank {self.local_rank}")
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

            sampler = EvalDistributedSampler(
                test_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,  # Usually False for evaluation/testing
            )
        else:
            sampler = None

        # Create dataloader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            num_workers=self.num_dataloader_workers,
            sampler=sampler,
            collate_fn=self.vlm_model.collate_fn,
        )

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        logger.info_main("Running inference...")

        results = []
        with torch.inference_mode():
            # Iterate over the test dataset
            for i, inputs in tqdm(enumerate(test_loader), total=len(test_loader)):
                raw_samples = inputs["raw_samples"]
                del inputs["raw_samples"]
                inputs = extract_batch_prompt_only(inputs)
                inputs = inputs.to(device="cuda", dtype=torch.bfloat16)

                if "batch_count" in inputs:
                    del inputs["batch_count"]

                if self.is_distributed:
                    outputs = model.module.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                    )
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                    )

                trimmed_generated_ids = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
                ]
                decoded_output = self.vlm_model.tokenizer.batch_decode(
                    trimmed_generated_ids, skip_special_tokens=True
                )

                logger.info_main(f"First batch sample {i + 1}/{len(test_loader)}:\n{decoded_output[0]}")

                results.extend(
                    [
                        {
                            **{key: value[i] for key, value in raw_samples.items()},
                            "output_text": do,
                        }
                        for i, do in enumerate(decoded_output)
                    ]
                )

        for result in results:
            # Clean up the output text
            outputs_text = result["output_text"]
            if "assistant\n" in outputs_text:
                outputs_text = outputs_text.split("assistant\n")[-1].strip()
            if "system\n" in outputs_text:
                outputs_text = outputs_text.split("system\n")[-1].strip()
            result["output_text_clean"] = outputs_text

        if self.is_distributed:
            all_results = gather_inference_outputs(local_results=results, main_rank=0)
        else:
            all_results = results

        if is_main_process():
            results_filename = "inference_results.json"

            all_results.sort(key=lambda d: (d["image_id"], d["question"]))
            save_json(all_results, Path(self.output_dir, results_filename))

            logger.info_main(f"Total number of inference samples: {len(all_results)}")
            logger.info_main(f"Saved inference results to {str(Path(self.output_dir, results_filename))}")

        # Make sure that main process finishes writing before continuing
        if self.is_distributed:
            dist.barrier()

        log_banner("Successfully finished Inference")


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Vision Language Model Inference")
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a specific checkpoint directory to load."
    )

    group.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to a run directory containing checkpoint subdirectories "
             "(requires --mode to select which checkpoint to load)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["best_metric", "best", "final"],
        default="final",
        help="When using --run_dir, choose which checkpoint to load: 'best_metric', 'best' or 'final'."
    )

    parser.add_argument(
        "--inference_dir_suffix",
        type=str,
        default="",
        help="An optional suffix to append to the 'inference' output directory."
    )

    args, _ = parser.parse_known_args()
    return args


def main_cli():
    args = parse_args()
    main(args)


def main(args, is_main=True):
    seed_everything(0)  # Set a fixed seed for reproducibility

    if args.run_dir:
        run_dir = Path(args.run_dir)

        # If 'run_dir' does not exist, search for it in the output directory.
        if not run_dir.is_dir():
            run_dir = find_run_dir_in_output_dir(str(run_dir))

        vlm_model, cfg, ckpt_dir = AbstractVLMModel.load_from_run_dir(run_dir, args.mode)
    else:  # if args.checkpoint_dir
        ckpt_dir = args.checkpoint_dir
        vlm_model, cfg = AbstractVLMModel.load_from_checkpoint_dir(ckpt_dir)

    if is_main:  # we don't want to merge args if this was called by train.py
        cfg.merge_subconfigs_with_cli_args()

    inference_output_dir = Path(ckpt_dir, "inference" + getattr(args, "inference_dir_suffix", ""))

    log_file_name = add_datetime_tag("inference") + ".log"
    log_file_path = Path(inference_output_dir, log_file_name)
    setup_logger(log_file_path)

    if cfg.model.vision_encoder:
        vision_encoder_patch_size = cfg.model.vision_encoder.patch_size
    else:
        vision_encoder_patch_size = None

    logger.info_main(f"Using {cfg.eval.split} split for inference.")
    if cfg.data.dataset_name == "amosmm":
        test_dataset = AMOSMMDataset(
            dataset_json_path=cfg.eval.dataset_json_path,
            model_type=cfg.model.model_type,
            split=cfg.eval.split,
            vision_input_path=cfg.data.vision_input_path,
            vision_input_type=cfg.data.vision_input_type,
            answer_types=cfg.data.answer_types,
            n_samples=cfg.eval.n_samples,  # Limit the validation set size
            do_resize=cfg.data.do_resize,
            do_transforms=False,
            text_only=cfg.data.text_only,
            embedding_constant=cfg.data.embedding_constant,
            vision_encoder_patch_size=vision_encoder_patch_size,
            do_sentence_shuffle=False
        )
    elif cfg.data.dataset_name == "ctrate":
        test_dataset = CTRATEDataset(
            dataset_json_path=cfg.eval.dataset_json_path,
            model_type=cfg.model.model_type,
            split=cfg.eval.split,
            vision_input_path=cfg.data.vision_input_path,
            vision_input_type=cfg.data.vision_input_type,
            answer_types=cfg.data.answer_types,
            n_samples=cfg.eval.n_samples,  # Limit the validation set size
            do_resize=cfg.data.do_resize,
            do_transforms=False,
            text_only=cfg.data.text_only,
            embedding_constant=cfg.data.embedding_constant,
            vision_encoder_patch_size=vision_encoder_patch_size,
            do_sentence_shuffle=False,
            image_id_suffix=cfg.data.image_id_suffix
        )
    else:
        raise ValueError(f"Unsupported dataset name: {cfg.data.dataset_name}")

    logger.info_main(f"Loaded {len(test_dataset)} inference samples")

    if cfg.data.embedding_constant is not None:
        logger.info_main(f"Using embedding constant: {cfg.data.embedding_constant}")
        print(f"Using embedding constant: {cfg.data.embedding_constant}")

    # Create inference object and run inference
    inference = UnifiedVLMInference(vlm_model, cfg=cfg, output_dir=inference_output_dir,
                                    per_device_batch_size=cfg.eval.per_device_batch_size,
                                    num_dataloader_workers=cfg.eval.num_dataloader_workers,
                                    max_new_tokens=cfg.eval.max_new_tokens)

    inference.inference(test_dataset)
    logger.remove()

    if cfg.eval.run_eval and is_main_process():
        eval_args = argparse.Namespace(
            task_type="RG" if cfg.data.dataset_name == "ctrate" else "VQA",  # TODO: temporary hardcode
            inference_dir=inference_output_dir,
            num_processes=cfg.eval.num_eval_processes
        )
        eval_main(eval_args)


if __name__ == "__main__":
    main_cli()
