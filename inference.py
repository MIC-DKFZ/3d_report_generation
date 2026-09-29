import os
import sys
import argparse
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from vlm.configs.run_config import RunConfig
from vlm.data.datasets import (
        AMOSMMDataset,
        CTRATEDataset,
        MerlinDataset,
        MerlinDatasetRetrieval,
        MerlinDatasetRetrievalWithImage,
        CTRATEDatasetRetrieval,
        CTRATEDatasetRetrievalWithImage,
        BonnDataset,
        BonnDatasetRetrieval,
        BonnDatasetRetrievalWithImage,
        ToothFairy4Dataset,
)
from vlm.evaluation.eval_distributed_sampler import EvalDistributedSampler
from vlm.models.abstract_vl_wrapper import AbstractVLWrapper
from vlm.utils import logger
from vlm.utils.decoding import decode_generated_answers, decode_prompts
from vlm.utils.ddp_utils import setup_ddp, gather_inference_outputs, is_main_process
from vlm.utils.filter_batch import extract_batch_prompt_only
from vlm.utils.helper import add_datetime_tag, seed_everything
from vlm.utils.config.merging import recursive_dict_merge
from vlm.utils.load_and_save_files import load_config_file, save_json, save_yaml
from vlm.utils.logging import setup_logger, log_banner
from vlm.utils.paths import VLM_CONFIG_DIR, find_run_dir_in_output_dir
from vlm.utils.local_staging import get_local_vision_input_path
from eval import main as eval_main


class UnifiedVLMInference:
    def __init__(
        self,
        vlm_model: AbstractVLWrapper,
        cfg: RunConfig,
        output_dir: str | Path,
        per_device_batch_size: int,
        num_dataloader_workers: int,
        max_new_tokens: int,
    ):
        """
        Initialize the unified VLM inference.
        Args:
            vlm_model: An instance of a class derived from AbstractVLWrapper
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

        (
            self.is_distributed,
            self.rank,
            self.world_size,
            self.local_rank,
            self.local_world_size,
        ) = setup_ddp()

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
            pin_memory=False,
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

                decoded_output = decode_generated_answers(
                    self.vlm_model.tokenizer,
                    outputs,
                    inputs["input_ids"],
                )
                decoded_input = decode_prompts(
                    self.vlm_model.tokenizer,
                    inputs["input_ids"],
                )

                logger.info_main(
                    f"First batch sample {i + 1}/{len(test_loader)}:\n{decoded_output[0]}"
                )

                results.extend(
                    [
                        {
                            **{key: value[i] for key, value in raw_samples.items()},
                            "complete_input": decoded_input[i],
                            "output_text": do,
                            "output_text_clean": do.strip(),
                        }
                        for i, do in enumerate(decoded_output)
                    ]
                )

        if self.is_distributed:
            all_results = gather_inference_outputs(local_results=results, main_rank=0)
        else:
            all_results = results

        if is_main_process():
            results_filename = "inference_results.json"

            all_results.sort(key=lambda d: (d["image_id"], d["question"]))
            save_json(all_results, Path(self.output_dir, results_filename))

            logger.info_main(f"Total number of inference samples: {len(all_results)}")
            logger.info_main(
                f"Saved inference results to {str(Path(self.output_dir, results_filename))}"
            )

        # Make sure that main process finishes writing before continuing
        if self.is_distributed:
            dist.barrier()

        log_banner("Successfully finished Inference")


def parse_args() -> tuple[Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Vision Language Model Inference", allow_abbrev=False
    )
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a specific checkpoint directory to load.",
    )

    group.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to a run directory containing checkpoint subdirectories "
        "(requires --mode to select which checkpoint to load).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["best", "final"],
        default="final",
        help="When using --run_dir, choose which checkpoint to load: 'best' or 'final'.",
    )

    parser.add_argument(
        "--inference_dir_suffix",
        type=str,
        default="",
        help="An optional suffix to append to the 'inference' output directory."
    )

    parser.add_argument(
        "--is_staged",
        action="store_true",
        help=(
            "Use the node-local staged vision input directory returned by "
            "get_local_vision_input_path()."
        ),
    )
    parser.add_argument(
        "--overwrite_config_from_files",
        type=str,
        nargs="+",
        default=[],
        help="Optional list of base config files",
    )

    return parser.parse_known_args()


def main_cli():
    args, config_cli_args = parse_args()
    main(args, config_cli_args=config_cli_args)


def main(
    args,
    vision_input_path: str | None = None,
    is_main: bool = True,
    config_cli_args: list[str] | None = None,
):
    seed_everything(0)  # Set a fixed seed for reproducibility

    if args.run_dir:
        run_dir = Path(args.run_dir)

        # If 'run_dir' does not exist, search for it in the output directory.
        if not run_dir.is_dir():
            run_dir = find_run_dir_in_output_dir(str(run_dir))

        vlm_model, cfg, ckpt_dir = AbstractVLWrapper.load_from_run_dir(run_dir, args.mode)
    else:  # if args.checkpoint_dir
        ckpt_dir = args.checkpoint_dir
        vlm_model, cfg = AbstractVLWrapper.load_from_checkpoint_dir(ckpt_dir)

    # Moved this bevor `is_main`
    inference_output_dir = Path(ckpt_dir, "inference" + getattr(args, "inference_dir_suffix", ""))

    log_file_name = add_datetime_tag("inference") + ".log"
    log_file_path = Path(inference_output_dir, log_file_name)
    setup_logger(log_file_path)

    if is_main:  # we don't want to merge args if this was called by train.py
        if getattr(args, "overwrite_config_from_files", None):
            cfg_dict = cfg.model_dump()
            for override_file in args.overwrite_config_from_files:
                override_path = Path(VLM_CONFIG_DIR, override_file)
                override_cfg_dict = load_config_file(override_path)
                override_cfg_dict = RunConfig.apply_base_configs(
                    cfg_dict=override_cfg_dict, config_dir=VLM_CONFIG_DIR
                )
                cfg_dict = recursive_dict_merge(cfg_dict, override_cfg_dict)

            cfg = RunConfig.model_validate(cfg_dict)
            logger.info_main(f"Applied overrides from: {args.overwrite_config_from_files}")

        # CLI values take precedence over file overrides.
        cfg.merge_subconfigs_with_cli_args(config_cli_args or [])

    if getattr(args, "is_staged", False):
        cfg.data.vision_input_path = str(get_local_vision_input_path())

    cfg.data.vision_input_path = vision_input_path if vision_input_path else cfg.data.vision_input_path
    vision_encoder_patch_size = cfg.model.vision_encoder.patch_size if cfg.model.vision_encoder else None

    logger.info_main(f"Using {cfg.eval.split} split for inference.")
    cfg.data.do_sentence_shuffle = False
    cfg.data.do_transforms = False
    if cfg.data.dataset_name == "amosmm":
        test_dataset = AMOSMMDataset(
            cfg=cfg,
            split=cfg.eval.split,
            vision_encoder_patch_size=vision_encoder_patch_size,
        )
    elif cfg.data.dataset_name == "ctrate":
        if cfg.data.retrieval_json_path is not None:
            if cfg.data.retrieval_use_image_in_prompt:
                ctrate_dataset_class = CTRATEDatasetRetrievalWithImage
            else:
                ctrate_dataset_class = CTRATEDatasetRetrieval
        else:
            ctrate_dataset_class = CTRATEDataset

        test_dataset = ctrate_dataset_class(  # type: ignore
            cfg=cfg,
            split=cfg.eval.split,
            n_samples=None,
            vision_encoder_patch_size=vision_encoder_patch_size
        )
    elif cfg.data.dataset_name == "merlin":
        if cfg.data.retrieval_json_path is not None:
            if cfg.data.retrieval_use_image_in_prompt:
                merlin_dataset_class = MerlinDatasetRetrievalWithImage
            else:
                merlin_dataset_class = MerlinDatasetRetrieval
        else:
            merlin_dataset_class = MerlinDataset

        test_dataset = merlin_dataset_class(  # type: ignore
            cfg=cfg,
            split=cfg.eval.split,
            n_samples=None,
            vision_encoder_patch_size=vision_encoder_patch_size
        )
    elif cfg.data.dataset_name == "bonn":
        if cfg.data.retrieval_json_path is not None:
            if cfg.data.retrieval_use_image_in_prompt:
                bonn_dataset_class = BonnDatasetRetrievalWithImage
            else:
                bonn_dataset_class = BonnDatasetRetrieval
        else:
            bonn_dataset_class = BonnDataset

        test_dataset = bonn_dataset_class(
            cfg=cfg,
            split=cfg.eval.split,
            vision_encoder_patch_size=vision_encoder_patch_size,
        )
    elif cfg.data.dataset_name == "toothfairy4":
        test_dataset = ToothFairy4Dataset(
            cfg=cfg,
            split=cfg.eval.split,
            vision_encoder_patch_size=vision_encoder_patch_size,
        )
    else:
        raise ValueError(f"Unsupported dataset name: {cfg.data.dataset_name}")

    logger.info_main(f"Loaded {len(test_dataset)} inference samples")

    if cfg.data.embedding_constant is not None:
        logger.info_main(f"Using embedding constant: {cfg.data.embedding_constant}")

    # Create inference object and run inference
    logger.info_main(f"Config:\n{cfg}")
    inference = UnifiedVLMInference(
        vlm_model,
        cfg=cfg,
        output_dir=inference_output_dir,
        per_device_batch_size=cfg.eval.per_device_batch_size,
        num_dataloader_workers=cfg.eval.num_dataloader_workers,
        max_new_tokens=cfg.eval.max_new_tokens,
    )

    inference.inference(test_dataset)
    logger.remove()

    # Save configuration used for inference
    # Currently mostly dataset related configs could change - e.g. to evaluate OOD data etc
    if is_main_process():
        final_config_path = os.path.join(inference_output_dir, "final_config.yml")
        save_yaml(
            {
                "command": " ".join(sys.argv),
                "run": cfg.model_dump()
            },
            final_config_path,
            sort_keys=False
        )

    if cfg.eval.run_eval and is_main_process():
        eval_args = argparse.Namespace(
            dataset_name = cfg.data.dataset_name,
            task_type="RG" if cfg.data.dataset_name == "ctrate" else "VQA",  # TODO: temporary hardcode
            inference_dir=inference_output_dir,
            num_processes=cfg.eval.num_eval_processes
        )
        eval_main(eval_args)


if __name__ == "__main__":
    main_cli()
