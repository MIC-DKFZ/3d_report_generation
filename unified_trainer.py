import argparse
import gc
import os
import random
import time
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from owl.callbacks import PrintEvalBatchCallback
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

import wandb
from data.datasets import AMOSMMDataset
from models.owl_vlm import OwlVLModel
from models.qwen_vlm import QwenVLModel
from unified_trainer import UnifiedVLMTrainer


class UnifiedVLMTrainer:
    def __init__(self, vlm_model, output_dir="vlm_output"):
        """
        Initialize the unified VLM trainer.

        Args:
            vlm_model: An instance of a class derived from AbstractVLMModel
            output_dir: Directory to save model outputs
        """
        self.vlm_model = vlm_model
        self.output_dir = output_dir
        self.trainer = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def get_training_args(self):
        """Configure standard training arguments"""
        base_args = {
            "output_dir": self.output_dir,
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
            "run_name": f"{type(self.vlm_model).__name__}_training",
        }

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
        if trainable_params.get("train_only_vision_adapter", False) == False:
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

        # Initialize wandb
        wandb.init(project="vlm_unified", name=f"{type(self.vlm_model).__name__}_run", reinit=True)

        # Configure training arguments
        training_args = self.get_training_args()

        # Create the callback
        callback = PrintEvalBatchCallback(tokenizer=self.vlm_model.tokenizer, processor=self.vlm_model.processor)

        # Initialize SFT trainer
        self.trainer = SFTTrainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.vlm_model.collate_fn,
            tokenizer=self.vlm_model.tokenizer,
            callbacks=[callback],
            # No need to pass peft_config here, we've already applied it
        )

        # Train model
        self.trainer.train()

        # Save model
        self.trainer.save_model(self.output_dir)

        # Clear memory
        self.vlm_model.clear_memory()
        del self.trainer

        # Finish wandb run
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Language Model")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["owl", "qwen"],
        required=True,
        help="Type of VLM model to train (owl or qwen)",
    )
    parser.add_argument("--model_id", type=str, required=True, help="Model ID or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="vlm_output", help="Directory to save model outputs")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to the vision embeddings")
    parser.add_argument(
        "--train_only_vision_adapter", action="store_true", help="Train only the vision adapter (Qwen only)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create the appropriate VLM model based on model_type
    if args.model_type == "owl":
        vlm_model = OwlVLModel(model_id=args.model_id, output_dir=args.output_dir)
    elif args.model_type == "qwen":
        vlm_model = QwenVLModel(
            model_id=args.model_id,
            output_dir=args.output_dir,
            train_only_vision_adapter=args.train_only_vision_adapter,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Create datasets
    train_dataset = AMOSMMDataset(
        dataset_json_path=args.dataset_path,
        model_name=args.model_type,
        split="train",
        embeddings_path=args.embeddings_path,
    )

    val_dataset = AMOSMMDataset(
        dataset_json_path=args.dataset_path,
        model_name=args.model_type,
        split="val",
        embeddings_path=args.embeddings_path,
        n_samples=50,  # Limit the validation set size
    )

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

    # Create the trainer and train the model
    trainer = UnifiedVLMTrainer(vlm_model, output_dir=args.output_dir)
    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
