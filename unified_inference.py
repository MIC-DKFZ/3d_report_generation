import argparse
import json
import os

import torch
import yaml

from data.datasets import AMOSMMDataset
from models.owl_vlm import OwlVLModel
from models.qwen_vlm import QwenVLModel


class UnifiedVLMInference:
    def __init__(
        self, vlm_model, output_dir="vlm_inference", config_file_path: str = None
    ):
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
        print("Running inference...")
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

        with torch.no_grad():
            # Iterate over the test dataset
            for i, inputs in enumerate(test_loader):
                inputs = inputs.to("cuda")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,  # Adjust max_new_tokens as needed
                )
                trimmed_generated_ids = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs["input_ids"], outputs)
                ]
                decoded_output = self.vlm_model.tokenizer.batch_decode(
                    trimmed_generated_ids, skip_special_tokens=True
                )
                decoded_output_non_trimmed = self.vlm_model.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                input_text = self.vlm_model.tokenizer.batch_decode(
                    inputs["input_ids"], skip_special_tokens=True
                )
                print(decoded_output)
                print(decoded_output_non_trimmed)
                print(input_text)


def parse_args(default_config=None):
    if default_config is None:
        default_config = {}
    general_parser = argparse.ArgumentParser(
        description="Train a Vision Language Model"
    )
    general_parser.add_argument(
        "--model_type",
        type=str,
        choices=["owl", "qwen"],
        required=True,
        help="Type of VLM model to train (owl or qwen)",
    )
    general_parser.add_argument(
        "--model_id", type=str, required=False, help="Model ID or path"
    )
    general_parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset JSON file"
    )
    general_parser.add_argument(
        "--output_dir",
        type=str,
        default="vlm_inference",
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
    print(f"Loaded {len(test_dataset)} test samples")

    # Create inference object and run inference
    inference = UnifiedVLMInference(vlm_model, output_dir=args.output_dir)
    inference.inference(test_dataset)


if __name__ == "__main__":
    main()
