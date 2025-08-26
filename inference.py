import argparse
import json
import os
import re
from datetime import datetime
from uuid import uuid4

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.datasets import AMOSMMDataset, CTRATEDataset, CTRATEDataset_Submission
from evaluation.metrics import bleu_score, meteor_score, rouge_score
from models.owl_vlm import OwlVLModel
from models.qwen_vlm import QwenVLModel
from utils import logger, rank_zero_filter
from utils.filter_batch import extract_batch_prompt_only


class UnifiedVLMInference:
    def __init__(self, vlm_model, output_dir="vlm_inference", config_file_path: str = None):
        """
        Initialize the unified VLM inference."
        Args:
            vlm_model: An instance of a class derived from AbstractVLMModel
            output_dir: Directory to save inference outputs
        """
        self.vlm_model = vlm_model
        self.output_dir = output_dir
        self.config_file_path = config_file_path

        # self.model_str = self.vlm_model._model_str  # e.g., "owl" or "qwen"
        self.model_str = "owl" if isinstance(self.vlm_model, OwlVLModel) else "qwen"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

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

    def inference(self, test_dataset, challenge_submission=False):
        """
        Main inference function that runs inference on the VLM model.
        Args:
            test_dataset: Test dataset
        """
        logger.info("Running inference...")
        model = self.vlm_model.setup_model()
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
                            max_new_tokens=256,  # Adjust max_new_tokens as needed
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
            if self.model_str == "owl" and "'text':" in outputs_text:
                outputs_text = re.search(r"'text':\s*'([^']*)'", outputs_text).group(1)
            result["output_text_clean"] = outputs_text

        preds = [result["output_text_clean"] for result in results]
        gts = [result["answer"] for result in results]

        if challenge_submission:
            results_report = [r for r in results if r["answer_subtype"] == "REPORT"]
            results_diagnosis = [r for r in results if r["answer_subtype"] == "DIAGNOSIS"]
            results_concat = []
            image_ids = set([r["image_id"] for r in results])
            for image_id in image_ids:
                if image_id not in [r["image_id"] for r in results_report]:
                    logger.warning(f"Image ID {image_id} has no report generation result.")
                    finding = [r for r in results_report if r["image_id"] == image_id]
                    if len(finding) > 1:
                        logger.warning(
                            f"Multiple report results for image ID {image_id}. Using the first one."
                        )
                    results_concat.append(
                        {
                            "input_image_name": image_id,
                            "report": f"Findings: {finding[0]['output_text_clean']}",
                        }
                    )
                if image_id not in [r["image_id"] for r in results_diagnosis]:
                    logger.warning(f"Image ID {image_id} has no diagnosis generation result.")
                    impression = [r for r in results_diagnosis if r["image_id"] == image_id]
                    if len(impression) > 1:
                        logger.warning(
                            f"Multiple diagnosis results for image ID {image_id}. Using the first one."
                        )
                    results_concat.append(
                        {
                            "input_image_name": image_id,
                            "report": f"Impression: {impression[0]['output_text_clean']}",
                        }
                    )
                else:
                    finding = [r for r in results_report if r["image_id"] == image_id]
                    if len(finding) > 1:
                        logger.warning(
                            f"Multiple diagnosis results for image ID {image_id}. Using the first one."
                        )
                    impression = [r for r in results_diagnosis if r["image_id"] == image_id]
                    if len(impression) > 1:
                        logger.warning(
                            f"Multiple report results for image ID {image_id}. Using the first one."
                        )
                    results_concat.append(
                        {
                            "input_image_name": image_id,
                            "report": f"Findings: {finding[0]['output_text_clean']}\nImpression: {impression[0]['output_text_clean']}",
                        }
                    )
            results = {
                "name": "Generated reports",
                "type": "Report generation",
                "generated_reports": results_concat,
                "version": {"major": 1, "minor": 0},
            }

        results_filename = "inference_results.json" if not challenge_submission else "results.json"

        with open(os.path.join(self.output_dir, results_filename), "w") as f:
            json.dump(results, f, indent=4)

        logger.info(
            f"Saved inference results to {os.path.join(self.output_dir, results_filename)}"
        )

        if not challenge_submission:
            # Calculate and log metrics
            # Log the Bleu score
            bleu_score_1 = bleu_score(preds, gts, max_order=1)["bleu"]
            bleu_score_2 = bleu_score(preds, gts, max_order=2)["bleu"]
            bleu_score_3 = bleu_score(preds, gts, max_order=3)["bleu"]
            bleu_score_4 = bleu_score(preds, gts, max_order=4)["bleu"]
            logger.info(f"Bleu1 score: {bleu_score_1}")
            logger.info(f"Bleu2 score: {bleu_score_2}")
            logger.info(f"Bleu3 score: {bleu_score_3}")
            logger.info(f"Bleu4 score: {bleu_score_4}")

            # Log the METOR score
            meteor_score_val = meteor_score(preds, gts)["meteor"]
            logger.info(f"METEOR score: {meteor_score_val}")

            # Log the ROUGE score
            rouge_score_val = rouge_score(preds, gts)["rougeL"]
            logger.info(f"ROUGE score: {rouge_score_val}")

            with open(os.path.join(self.output_dir, "inference_metrics.json"), "w") as f:
                json.dump(
                    {
                        "bleu_score_1": bleu_score_1,
                        "bleu_score_2": bleu_score_2,
                        "bleu_score_3": bleu_score_3,
                        "bleu_score_4": bleu_score_4,
                        "meteor_score": meteor_score_val,
                        "rouge_score": rouge_score_val,
                    },
                    f,
                    indent=4,
                )

            logger.info(
                f"Saved inference metrics to {os.path.join(self.output_dir, 'inference_metrics.json')}"
            )


def parse_args(default_config=None):
    if default_config is None:
        default_config = {}
    general_parser = argparse.ArgumentParser(description="Train a Vision Language Model")
    general_parser.add_argument(
        "--model_type",
        type=str,
        choices=["owl", "qwen"],
        required=True,
        help="Type of VLM model to train (owl or qwen)",
    )
    general_parser.add_argument("--model_id", type=str, required=False, help="Model ID or path")
    # general_parser.add_argument(
    #     "--base_model_id",
    #     type=str,
    #     required=False,
    #     help="Base Model ID or path. Only needed for Qwen if model_id is a fine-tuned checkpoint path",
    # )
    general_parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset JSON file"
    )
    general_parser.add_argument(
        "--vision_encoder_path", type=str, required=False, help="Path to the dataset JSON file"
    )
    general_parser.add_argument(
        "--output_dir",
        type=str,
        default="output/vlm_inference",
        help="Directory to save model outputs",
    )
    general_parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to the vision embeddings",
    )

    general_parser.add_argument(
        "--embedding_constant",
        # type=float,
        default=None,
        help="Constant value for training language only baseline (! Do not set this when training a VLM !). Can either be a float to set the whole embedding to a constant value or 'random' to set the embedding to a random values.",
    )
    general_parser.add_argument(
        "--include_cls_text_gen",
        action="store_true",
        help="Include classes from classification for text generation (report/diagnosis) (CTRATE only)",
    )
    general_parser.add_argument(
        "--cls_aug_strength",
        type=float,
        default=0.0,
        help="Strength of the class augmentation for text generation, meaning probability of flipping labels for individual classes. E.g. 0.05 means 5 percent of the time a class label will be flipped to another class label.  Only used for training dataset. (CTRATE only)",
    )
    general_parser.add_argument(
        "--inference_config", type=str, help="Path to config file (YAML or JSON)"
    )
    general_parser.add_argument(
        "--num_gpus", type=str, help="Number of GPUs to use for inference", default=1
    )
    general_parser.add_argument(
        "--is_base_model",
        action="store_true",
        help="Set this flag for Owl model if you want to run inference on the base model, otherwise it will be assumed that model_id is a fine-tuned checkpoint path",
    )
    general_parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to a model checkpoint to load"
    )
    general_parser.add_argument(
        "--dataset_name", type=str, default="ctrate", help="Name of the dataset"
    )
    general_parser.add_argument(
        "--dataset_cache_val", type=str, default=None, help="Path to cache file for the dataset"
    )
    general_parser.add_argument(
        "--load_image",
        action="store_true",
        default=False,
        help="Load image data during inference (default: False)",
    )
    general_parser.add_argument(
        "--use_csv_reports",
        action="store_true",
        help="Get reports from CSV file instead of yml files",
    )
    general_parser.add_argument(
        "--do_resize",
        action="store_true",
        help="Resize the input image to 192x192x192",
    )
    general_parser.add_argument(
        "--do_training_transforms",
        action="store_true",
        help="Wheather or not to do data augmentation to the image during training",
    )
    general_parser.add_argument(
        "--do_validation_transforms",
        action="store_true",
        help="Wheather or not to do data augmentation to the image during training",
    )
    
    # general_parser.add_argument(
    #     "--no_calculate_metrics",
    #     action="store_true",
    #     help="Not calculate and store metrics",
    # )
    general_parser.add_argument(
        "--challenge_submission",
        action="store_true",
        help="Set this flag if you want to run inference for challenge submission (no metrics calculation, different output format)",
    )

    # First parse just the config argument to load the file
    general_args, _ = general_parser.parse_known_args()

    # Load config file if provided
    if general_args.inference_config and os.path.exists(general_args.inference_config):
        file_ext = os.path.splitext(general_args.inference_config)[1].lower()
        with open(general_args.inference_config, "r") as file:
            if file_ext == ".yaml" or file_ext == ".yml":
                file_config = yaml.safe_load(file)
            elif file_ext == ".json":
                file_config = json.load(file)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")

            # Update config with values from file
            default_config.update(file_config)
    if general_args.embedding_constant is not None:
        try:
            general_args.embedding_constant = float(general_args.embedding_constant)
        except ValueError:
            pass

    return general_args


def main():
    args = parse_args()

    local_rank = 0
    if int(args.num_gpus) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    unique_id = str(uuid4()) if not slurm_job_id else slurm_job_id
    log_file_name = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_inference_{unique_id}.log")
    log_file = os.path.join(args.output_dir, log_file_name)
    if local_rank == 0:
        logger.add(log_file, filter=rank_zero_filter)
    if args.model_id is None:
        if args.model_type == "owl":
            args.model_id = "mPLUG/mPLUG-Owl3-1B-241014"
        elif args.model_type == "qwen":
            args.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Create the appropriate VLM model based on the model type
    model_specific_args = {}
    if args.model_type == "owl":
        vlm_model_class = OwlVLModel
        model_specific_args = {"checkpoint_path": args.checkpoint_path}
    elif args.model_type == "qwen":
        vlm_model_class = QwenVLModel
        model_specific_args = {"checkpoint_path": args.checkpoint_path, "vision_encoder_path": args.vision_encoder_path}
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    vlm_model = vlm_model_class(
        model_id=args.model_id,
        output_dir=args.output_dir,
        train_only_vision_adapter=False,
        **model_specific_args,
    )
    if args.dataset_name == "amosmm":
        test_dataset = AMOSMMDataset(
            dataset_json_path=args.dataset_path,
            model_name=args.model_type,
            split="val",
            embeddings_path=args.embeddings_path,
        )
        logger.info(f"Loaded {len(test_dataset)} test samples")
    elif args.dataset_name == "ctrate":
        if args.challenge_submission: 
            test_dataset = CTRATEDataset_Submission(
                dataset_json_path=args.dataset_path,
                model_name=args.model_type,
                split="test",
                embeddings_path=args.embeddings_path,
                # n_samples=args.n_test_samples,  # Limit the test set size
                # split_files=args.split_file,
                answer_types=("TEXT",),
                cache_file=args.dataset_cache_val,  # Use cache file if provided
            )
        else:
            if args.include_cls_text_gen:
                logger.info(f"Including classes in text generation for CTRate with augmentation strength {args.cls_aug_strength}")
            test_dataset = CTRATEDataset(
                dataset_json_path=args.dataset_path,
                model_name=args.model_type,
                split="val",
                embeddings_path=args.embeddings_path,
                use_csv_reports=args.use_csv_reports,
                # n_samples=args.n_val_samples,  # Limit the validation set size
                # split_files=args.split_file,
                answer_types=("TEXT",),
                include_cls_text_gen=args.include_cls_text_gen,
                cls_aug_strength=args.cls_aug_strength,
                cache_file=args.dataset_cache_val,  # Use cache file if provided
                load_image=args.load_image,
                do_resize=args.do_resize,
                do_transforms=args.do_validation_transforms
            )
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")
    if args.embedding_constant is not None:
        logger.info(f"Setting embedding constant to {args.embedding_constant}")
        print(f"Setting embedding constant to {args.embedding_constant}")
        test_dataset.embedding_constant = args.embedding_constant

    # Create inference object and run inference
    inference = UnifiedVLMInference(vlm_model, output_dir=args.output_dir)
    # calculate_metrics = not args.no_calculate_metrics
    inference.inference(test_dataset, challenge_submission=args.challenge_submission)


if __name__ == "__main__":
    main()
