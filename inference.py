import argparse
import os
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from configs.run_config import RunConfig
from data.datasets import AMOSMMDataset, CTRATEDataset
from eval_ctrate import main as eval_ctrate
from evaluation.metrics import bleu_score, meteor_score, rouge_score
from models.abstract_vlm import AbstractVLMModel
from utils import logger, rank_zero_filter
from utils.filter_batch import extract_batch_prompt_only
from utils.helper import add_datetime_tag, is_main_process
from utils.load_and_save_files import load_json, save_json
from utils.vision_input_type import VisionInputType


class UnifiedVLMInference:
    def __init__(self, vlm_model: AbstractVLMModel, cfg: RunConfig, output_dir: str | Path):
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

        # TODO optionally load those member variables from config file
        self.num_dataloader_workers = 4
        self.per_device_eval_batch_size = 1

        self.ddp_available = dist.is_available() and dist.is_initialized()
        if self.ddp_available:
            logger.info("Distributed training is available and initialized.")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.global_rank = int(os.environ["RANK"])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            logger.info("Distributed training is not available or not initialized.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.local_rank = 0  # or handle as single process
            self.world_size = 1

    def inference(self, test_dataset):
        """
        Main inference function that runs inference on the VLM model.
        Args:
            test_dataset: Test dataset
        """
        logger.info("Running inference...")
        model = self.vlm_model.model
        model.eval().cuda()

        if self.ddp_available:
            # Wrap the model with DDP
            logger.info(f"Wrapping model with DDP on rank {self.local_rank}")
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

            sampler = DistributedSampler(
                test_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=False,  # Usually False for evaluation/testing
            )
        else:
            sampler = None

        # Create dataloader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sampler=sampler,
            collate_fn=self.vlm_model.collate_fn,
        )

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        results = []
        with torch.no_grad():
            # Iterate over the test dataset
            for i, inputs in tqdm(enumerate(test_loader), total=len(test_loader)):
                raw_samples = inputs["raw_samples"]
                del inputs["raw_samples"]
                inputs = extract_batch_prompt_only(inputs)
                inputs = inputs.to("cuda")

                if "batch_count" in inputs:
                    del inputs["batch_count"]

                # image_embeds is only in inputs for owl model due to cross attn
                if "image_embeds" in inputs:
                    if self.ddp_available:
                        inputs["image_embeds"] = model.module.vision2text_model(
                            inputs["image_embeds"]
                        )
                        outputs = model.module.language_model.generate(
                            **inputs,
                            max_new_tokens=1024,  # Adjust max_new_tokens as needed
                        )
                    else:
                        inputs["image_embeds"] = model.vision2text_model(inputs["image_embeds"])
                        outputs = model.language_model.generate(
                            **inputs,
                            max_new_tokens=1024,  # Adjust max_new_tokens as needed
                        )

                # otherwise, image embeddings are already contained in input_ids
                else:
                    if self.ddp_available:
                        outputs = model.module.generate(
                            **inputs,
                            max_new_tokens=1024,  # Adjust max_new_tokens as needed
                        )
                    else:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1024,  # Adjust max_new_tokens as needed
                        )
                trimmed_generated_ids = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
                ]
                decoded_output = self.vlm_model.tokenizer.batch_decode(
                    trimmed_generated_ids, skip_special_tokens=True
                )

                # raw_input

                # Log the first sample output
                if i == 0:
                    logger.info(f"First sample input: {raw_samples['answer'][0]}")
                    logger.info(f"First sample output: {decoded_output[0]}")

                logger.info(f"Sample {i + 1}/{len(test_loader)}: {decoded_output}")

                results.extend(
                    [
                        {
                            **{key: value[i] for key, value in raw_samples.items()},
                            "output_text": do,
                        }
                        for i, do in enumerate(decoded_output)
                    ]
                )

                # if i == 50:
                #     logger.info("Stopping after 50 samples for testing purposes.")
                #     break

        for result in results:
            # Clean up the output text
            outputs_text = result["output_text"]
            if "assistant\n" in outputs_text:
                outputs_text = outputs_text.split("assistant\n")[-1].strip()
            if "system\n" in outputs_text:
                outputs_text = outputs_text.split("system\n")[-1].strip()
            result["output_text_clean"] = outputs_text

        results_filename = "inference_results.json"

        save_json(results, Path(self.output_dir, results_filename))

        logger.info(
            f"Saved inference results to {str(Path(self.output_dir, results_filename))}"
        )


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
        choices=["best", "final"],
        default="best",
        help="When using --run_dir, choose which checkpoint to load: 'best' or 'final'."
    )
    parser.add_argument(
        "--test_dataset_json",
        type=str,
        required=True,
        help="Path to the testset dataset JSON file."
    )

    parser.add_argument(
        "--vision_encoder_path",
        type=str,
        default=None,
        help="Path to the vision encoder checkpoint."
    )
    parser.add_argument(
        "--vision_input_path",
        type=str,
        default=None,
        help="Path to the vision input directory."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples. Run all if 'None'."
    )
    parser.add_argument(
        "--final_test",
        action="store_true",
        help="If set, set split to test instead of val."
    )

    # First parse just the config argument to load the file
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    test_dataset_json_path = Path(args.test_dataset_json)
    if not test_dataset_json_path.exists():
        raise FileNotFoundError(f"Test dataset JSON not found: {str(test_dataset_json_path)}")

    if args.run_dir:
        vlm_model, cfg, ckpt_dir = AbstractVLMModel.load_from_run_dir(args.run_dir, args.mode)
    else:  # if args.checkpoint_dir
        ckpt_dir = args.checkpoint_dir
        vlm_model, cfg = AbstractVLMModel.load_from_checkpoint_dir(ckpt_dir)

    if args.vision_encoder_path:
        cfg.model.vision_encoder.vision_encoder_path = args.vision_encoder_path
    if args.vision_input_path:
        cfg.data.vision_input_path = args.vision_input_path

    inference_output_dir = ckpt_dir / "inference"

    local_rank = 0
    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    log_file_name = add_datetime_tag("inference.log")
    log_file = Path(inference_output_dir, log_file_name)
    if local_rank == 0:
        logger.add(log_file, filter=rank_zero_filter)

    if cfg.data.vision_input_type == VisionInputType.IMAGE:
        ve_plans = load_json(Path(cfg.model.vision_encoder.vision_encoder_path, "plans.json"))
        vision_encoder_patch_size = ve_plans["configurations"]["3d_fullres"]["patch_size"]
    else:
        vision_encoder_patch_size = None

    split = "test" if args.final_test else "val"
    logger.info(f"Using {split} split for inference.")
    if cfg.data.dataset_name == "amosmm":
        test_dataset = AMOSMMDataset(
            dataset_json_path=str(test_dataset_json_path),
            model_name=cfg.model.model_type,
            split=split,
            vision_input_path=cfg.data.vision_input_path,
            vision_input_type=cfg.data.vision_input_type,
            answer_types=cfg.data.answer_types,
            n_samples=args.n_samples,  # Limit the validation set size
            do_resize=cfg.data.do_resize,
            do_transforms=cfg.data.do_transforms,
            embedding_constant=cfg.data.embedding_constant,
            vision_encoder_patch_size=vision_encoder_patch_size
        )
    elif cfg.data.dataset_name == "ctrate":
        test_dataset = CTRATEDataset(
            dataset_json_path=str(test_dataset_json_path),
            model_name=cfg.model.model_type,
            split=split,
            vision_input_path=cfg.data.vision_input_path,
            vision_input_type=cfg.data.vision_input_type,
            answer_types=cfg.data.answer_types,
            n_samples=args.n_samples,  # Limit the validation set size
            do_resize=cfg.data.do_resize,
            do_transforms=cfg.data.do_transforms,
            embedding_constant=cfg.data.embedding_constant,
            vision_encoder_patch_size=vision_encoder_patch_size
        )
    else:
        raise ValueError(f"Unsupported dataset name: {cfg.data.dataset_name}")

    logger.info(f"Loaded {len(test_dataset)} test samples")

    if cfg.data.embedding_constant is not None and is_main_process():
        logger.info(f"Using embedding constant: {cfg.data.embedding_constant}")
        print(f"Using embedding constant: {cfg.data.embedding_constant}")

    # Create inference object and run inference
    inference = UnifiedVLMInference(vlm_model, cfg=cfg, output_dir=inference_output_dir)
    # calculate_metrics = not args.no_calculate_metrics
    inference.inference(test_dataset)


if __name__ == "__main__":
    main()
