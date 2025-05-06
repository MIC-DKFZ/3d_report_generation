import argparse
import json
import os
from datetime import datetime
from uuid import uuid4

import torch
import yaml
from tqdm import tqdm

from data.datasets import AMOSMMDataset
from models.owl_vlm import OwlVLModel
from models.qwen_vlm import QwenVLModel
from utils import logger
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

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def inference(self, test_dataset):
        """
        Main inference function that runs inference on the VLM model.
        Args:
            test_dataset: Test dataset
        """
        logger.info("Running inference...")
        model = self.vlm_model.setup_model()
        model.eval().cuda()

        # Create dataloader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
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
                    inputs["image_embeds"] = model.vision2text_model(inputs["image_embeds"])
                    outputs = model.language_model.generate(
                        **inputs,
                        max_new_tokens=256,  # Adjust max_new_tokens as needed
                    )
                # otherwise, image embeddings are already contained in input_ids
                else:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,  # Adjust max_new_tokens as needed
                    )
                trimmed_generated_ids = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
                ]
                decoded_output = self.vlm_model.tokenizer.batch_decode(
                    trimmed_generated_ids, skip_special_tokens=True
                )

                results.extend(
                    [
                        {
                            **{key: value[i] for key, value in raw_samples.items()},
                            "output_text": do,
                        }
                        for i, do in enumerate(decoded_output)
                    ]
                )
        with open(os.path.join(self.output_dir, "inference_results.json"), "w") as f:
            json.dump(results, f, indent=4)


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
    general_parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset JSON file"
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
        "--inference_config", type=str, help="Path to config file (YAML or JSON)"
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

    return general_args


def main():
    args = parse_args()
    log_file = os.path.join(
        args.output_dir, datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_inference_{uuid4()}.log")
    )
    logger.add(log_file)
    if args.model_id is None:
        if args.model_type == "owl":
            args.model_id = "mPLUG/mPLUG-Owl3-1B-241014"
        elif args.model_type == "qwen":
            args.model_id = "Qwen/Qwen2-VL-2B-Instruct"

    # Create the appropriate VLM model based on the model type
    if args.model_type == "owl":
        vlm_model_class = OwlVLModel
    elif args.model_type == "qwen":
        vlm_model_class = QwenVLModel
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    vlm_model = vlm_model_class(
        model_id=args.model_id,
        output_dir=args.output_dir,
        train_only_vision_adapter=True,
    )

    test_dataset = AMOSMMDataset(
        dataset_json_path=args.dataset_path,
        model_name=args.model_type,
        split="val",
        embeddings_path=args.embeddings_path,
    )
    logger.info(f"Loaded {len(test_dataset)} test samples")

    # Create inference object and run inference
    inference = UnifiedVLMInference(vlm_model, output_dir=args.output_dir)
    inference.inference(test_dataset)


if __name__ == "__main__":
    main()
