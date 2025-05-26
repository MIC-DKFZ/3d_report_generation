import argparse
import gc
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

import wandb
from data.datasets import AMOSMMDataset
from models.owl_vlm import OwlVLModel
from models.qwen_vlm import QwenVLModel
from owl.callbacks import LossLoggingCallback, PrintEvalBatchCallback
from utils import logger
from utils.helper import seed_everything


class UnifiedVLMTrainer:
    def __init__(
        self,
        vlm_model,
        output_dir="vlm_output",
        config_file_path: str = None,
        sft_trainer_config: dict = None,
    ):
        """
        Initialize the unified VLM trainer.

        Args:
            vlm_model: An instance of a class derived from AbstractVLMModel
            output_dir: Directory to save model outputs
        """
        self.vlm_model = vlm_model
        self.output_dir = output_dir
        self.trainer = None
        self.config_file_path = config_file_path
        self.sft_trainer_config = sft_trainer_config

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def get_training_args(self):
        """Configure standard training arguments"""
        num_cores = os.cpu_count()
        num_workers = num_cores // 2 if num_cores > 4 else 1
        base_args = {
            "output_dir": self.output_dir,
            "run_name": f"{type(self.vlm_model).__name__}_training",
            "dataloader_num_workers": num_workers,
            "dataloader_pin_memory": True,
        }
        base_args.update(self.sft_trainer_config)

        # Add model-specific arguments
        model_args = self.vlm_model.get_model_specific_training_args()
        base_args.update(model_args)

        return SFTConfig(**base_args)

    def get_quantization_config(self):
        """Create and return the BitsAndBytes quantization config"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def apply_peft(self, base_model, peft_config):
        """Apply PEFT configuration to the model and return the PEFT model"""
        peft_model = get_peft_model(base_model, peft_config)
        peft_model.print_trainable_parameters()
        return peft_model

    def train(self, train_dataset, val_dataset):
        """
        Main training function that trains the VLM model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        # Get trainable parameters configuration
        trainable_params = self.vlm_model.get_trainable_params()

        # Create quantization config if needed (not for vision adapter only training)
        if trainable_params.get("train_only_vision_adapter", False) == True:
            bnb_config = self.get_quantization_config()
        else:
            bnb_config = None

        # Setup base model with quantization config if applicable
        base_model = self.vlm_model.setup_model(bnb_config=bnb_config)

        # Apply PEFT if specified in trainable_params
        peft_config = trainable_params.get("peft_config", None)
        if peft_config is not None:
            model_to_train = self.apply_peft(base_model, peft_config)
        else:
            model_to_train = self.vlm_model.model

        wandb.init(
            project="mic-rocket-vlm",
            name=f"{type(self.vlm_model).__name__}_run",
            reinit=True,
            group=f"{type(self.vlm_model).__name__}_run_{uuid4()}",
        )

        # Configure training arguments
        training_args = self.get_training_args()

        # Create the callback TODO implement this also for Qwen?
        if type(self.vlm_model).__name__ == "OwlVLModel":
            callbacks = [
                PrintEvalBatchCallback(
                    tokenizer=self.vlm_model.tokenizer,
                    processor=self.vlm_model.processor,
                ),
                LossLoggingCallback(),
            ]
        else:
            callbacks = [LossLoggingCallback()]

        # Initialize SFT trainer
        self.trainer = SFTTrainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.vlm_model.collate_fn,
            callbacks=callbacks,
            # No need to pass peft_config here, we've already applied it
        )

        # Train model
        self.trainer.train()

        # Save model
        self.trainer.save_model(self.output_dir)

        # Clear memory
        self.vlm_model.clear_memory()
        del self.trainer

        wandb.finish()


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
        default="output/vlm_output",
        help="Directory to save model outputs",
    )
    general_parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to the vision embeddings",
    )
    general_parser.add_argument(
        "--train_only_vision_adapter",
        action="store_true",
        help="Train only the vision adapter (Qwen only)",
    )
    general_parser.add_argument(
        "--train_config", type=str, help="Path to config file (YAML or JSON)"
    )

    # First parse just the config argument to load the file
    general_args, _ = general_parser.parse_known_args()

    sft_trainer_parser = argparse.ArgumentParser(
        description="SFTTrainer args which overwrite the train_config file."
    )

    # Load config file if provided
    if general_args.train_config and os.path.exists(general_args.train_config):
        file_ext = os.path.splitext(general_args.train_config)[1].lower()
        with open(general_args.train_config, "r") as file:
            if file_ext == ".yaml" or file_ext == ".yml":
                file_config = yaml.safe_load(file)
            elif file_ext == ".json":
                file_config = json.load(file)
            else:
                raise ValueError(f"Unsupported config file format: {file_ext}")

            # Update config with values from file
            default_config.update(file_config)

    # Add arguments for each config value with the default from config
    for key, value in default_config.items():
        arg_type = type(value)
        if arg_type == bool:
            # Handle boolean values with store_true/store_false
            if value:
                sft_trainer_parser.add_argument(
                    f"--no-{key}",
                    action="store_false",
                    dest=key,
                    help=f"Disable {key} (default: enabled)",
                )
            else:
                sft_trainer_parser.add_argument(
                    f"--{key}",
                    action="store_true",
                    help=f"Enable {key} (default: disabled)",
                )
        else:
            sft_trainer_parser.add_argument(
                f"--{key}",
                type=arg_type,
                default=value,
                help=f"Set {key} (default: {value})",
            )

    sft_trainer_parsed, _ = sft_trainer_parser.parse_known_args()
    return general_args, sft_trainer_parsed


def main():
    default_sft_trainer_config = {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "learning_rate": 2e-4,
        "lr_scheduler_type": "constant",
        "logging_steps": 10,
        "eval_steps": 50,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 2,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "load_best_model_at_end": True,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "report_to": ["wandb"],
    }
    general_args, sft_trainer_args = parse_args(default_sft_trainer_config)

    log_file = os.path.join(
        general_args.output_dir, datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_train_{uuid4()}.log")
    )
    logger.add(log_file)

    if general_args.model_id is None:
        if general_args.model_type == "owl":
            general_args.model_id = "mPLUG/mPLUG-Owl3-1B-241014"
        elif general_args.model_type == "qwen":
            general_args.model_id = "Qwen/Qwen2-VL-2B-Instruct"

    # Create the appropriate VLM model based on model_type
    if general_args.model_type == "owl":
        vlm_model_class = OwlVLModel
    elif general_args.model_type == "qwen":
        vlm_model_class = QwenVLModel
    else:
        raise ValueError(f"Unknown model type: {general_args.model_type}")

    seed_everything(42)  # Set a random seed for reproducibility

    vlm_model = vlm_model_class(
        model_id=general_args.model_id,
        output_dir=general_args.output_dir,
        train_only_vision_adapter=general_args.train_only_vision_adapter,
    )

    # Create datasets
    train_dataset = AMOSMMDataset(
        dataset_json_path=general_args.dataset_path,
        model_name=general_args.model_type,
        split="train",
        embeddings_path=general_args.embeddings_path,
    )

    val_dataset = AMOSMMDataset(
        dataset_json_path=general_args.dataset_path,
        model_name=general_args.model_type,
        split="val",
        embeddings_path=general_args.embeddings_path,
        n_samples=50,  # Limit the validation set size
    )
    logger.info(
        f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
    )

    # Create the trainer and train the model
    trainer = UnifiedVLMTrainer(
        vlm_model,
        output_dir=general_args.output_dir,
        sft_trainer_config=vars(sft_trainer_args),
    )
    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
