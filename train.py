import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
import torch.distributed as dist
import yaml
from peft import get_peft_model
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import wandb
from data.datasets import AMOSMMDataset, CTRATEDataset
from models.owl_vlm import OwlVLModel
from models.qwen_vlm import QwenVLModel
from owl.callbacks import LossLoggingCallback, PrintEvalBatchCallback
from utils import logger
from utils.helper import seed_everything
from utils.memory_utils import (
    aggressive_cleanup,
    log_memory_usage,
    setup_memory_efficient_training,
)
from utils.vision_input_type import VisionInputType


def setup_distributed_training():
    """
    Set up distributed training environment.
    This function handles the distributed training initialization as recommended in the JSC tutorial.
    """
    # Check if we're in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        logger.info(f"Distributed training detected: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        # Initialize the process group
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            logger.info(f"Process group initialized on rank {rank}")
        
        return True, rank, world_size, local_rank
    else:
        logger.info("Single GPU/CPU training detected")
        return False, 0, 1, 0


def is_main_process():
    """
    Check if this is the main process (rank 0).
    As recommended in the JSC tutorial, only rank 0 should handle logging, saving, etc.
    """
    return int(os.environ.get("RANK", "0")) == 0


class UnifiedVLMTrainer:
    def __init__(
        self,
        vlm_model,
        output_dir="vlm_output",
        config_file_path: str = None,
        sft_trainer_config: dict = None,
        wandb_project: str = "debug",
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
        self.wandb_project = wandb_project
        
        # Setup distributed training
        self.is_distributed, self.rank, self.world_size, self.local_rank = setup_distributed_training()

        # Create output directory if it doesn't exist (only on main process)
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

    def get_training_args(self):
        """Configure standard training arguments for distributed training"""
        num_cores = os.cpu_count()
        
        # Adjust number of workers based on distributed setup
        if self.is_distributed:
            # Reduce workers per process in distributed training
            num_workers = max(1, (num_cores // 4) // self.world_size)
        else:
            num_workers = 2
            
        base_args = {
            "output_dir": self.output_dir,
            "run_name": f"{type(self.vlm_model).__name__}_training",
            "dataloader_num_workers": num_workers,
            "dataloader_pin_memory": True,
            # Reduce prefetch factor to minimize memory usage
            "dataloader_prefetch_factor": 1,
            # Disable persistent workers to prevent memory accumulation
            "dataloader_persistent_workers": False,
            # Add torch empty cache steps to clear memory periodically
            "torch_empty_cache_steps": 20,
            # Distributed training specific settings
            "ddp_find_unused_parameters": False,  # Important for performance
            "ddp_bucket_cap_mb": 25,  # Reduce bucket size for better memory usage
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
        if is_main_process():
            peft_model.print_trainable_parameters()
        return peft_model

    def train(self, train_dataset, val_dataset):
        """
        Main training function that trains the VLM model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        # Setup memory efficient training
        setup_memory_efficient_training()
        if is_main_process():
            log_memory_usage("Before Training Setup")

        # Get trainable parameters configuration
        trainable_params = self.vlm_model.get_trainable_params()

        # Create quantization config if needed
        bnb_config = None
        # Uncomment if you want to use quantization
        # if trainable_params.get("train_only_vision_adapter", False) == True:
        #     bnb_config = self.get_quantization_config()
        # else:
        #     bnb_config = None

        # Setup base model with quantization config if applicable
        base_model = self.vlm_model.setup_model(bnb_config=bnb_config)

        # Apply PEFT if specified in trainable_params
        peft_config = trainable_params.get("peft_config", None)
        if peft_config is not None:
            model_to_train = self.apply_peft(base_model, peft_config)
        else:
            model_to_train = self.vlm_model.model

        # Create comprehensive config for WandB logging
        general_args = getattr(self, "general_args", None)
        dataset_info = getattr(self, "dataset_info", None)
        config_dict = create_comprehensive_wandb_config(
            self.vlm_model, self.sft_trainer_config, general_args, dataset_info, 
            self.world_size, self.rank
        )
        # Ensure trainable_params are included in the config
        config_dict["trainable_params"] = trainable_params

        # Only initialize wandb on the main process (rank 0) for DDP
        if is_main_process():
            wandb.init(
                entity="mic-rocket-vlm",
                project=self.wandb_project,
                name=datetime.now().strftime(
                    f"%Y-%m-%d_%H-%M-%S_train_{type(self.vlm_model).__name__}_run_{uuid4()}"
                ),
                reinit=True,
                group=f"{type(self.vlm_model).__name__}_run",
                config=config_dict,
            )

            # Ensure wandb properly logs hyperparameters in offline mode
            ensure_wandb_config_logging()

            # Additional explicit config update for offline mode robustness
            if os.getenv("WANDB_MODE", "online").lower() == "offline":
                # Re-update config one more time to ensure persistence
                wandb.config.update(config_dict, allow_val_change=True)
                logger.info("Offline mode: Config explicitly updated after init")
        else:
            logger.info(
                f"Skipping wandb init on rank {self.rank}/{self.local_rank}"
            )

        # Configure training arguments
        training_args = self.get_training_args()

        # Create callbacks (only add print callback on main process)
        callbacks = [LossLoggingCallback()]
        
        if is_main_process():
            callbacks.append(
                PrintEvalBatchCallback(
                    tokenizer=self.vlm_model.tokenizer,
                    processor=self.vlm_model.processor,
                    sample_count=len(val_dataset),
                )
            )

        # Initialize SFT trainer
        self.trainer = SFTTrainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.vlm_model.collate_fn,
            callbacks=callbacks,
        )

        # Train model
        if is_main_process():
            log_memory_usage("Before Training")
        
        self.trainer.train()
        
        if is_main_process():
            log_memory_usage("After Training")

        # Save model (only on main process)
        if is_main_process():
            self.trainer.save_model(self.output_dir)

        # Synchronize all processes before cleanup
        if self.is_distributed:
            dist.barrier()

        # Aggressive memory cleanup
        del self.trainer
        self.trainer = None

        # Clear model memory
        self.vlm_model.clear_memory()

        # Perform aggressive cleanup
        aggressive_cleanup()
        if is_main_process():
            log_memory_usage("After Cleanup")

        # Final config validation for offline mode
        if (is_main_process() and 
            os.getenv("WANDB_MODE", "online").lower() == "offline" and 
            wandb.run is not None):
            final_config = dict(wandb.config)
            logger.info(f"Final config verification - Total keys: {len(final_config)}")
            if len(final_config) < 5:  # Arbitrary threshold - expect at least some config
                logger.warning(
                    "Suspiciously few config keys logged - may indicate config persistence issue"
                )

        if is_main_process():
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
        "--train_dataset_json_path", type=str, required=True, help="Path to the train dataset JSON file"
    )
    general_parser.add_argument(
        "--val_dataset_json_path", type=str, required=False, help="Path to the validation dataset JSON file"
    )
    general_parser.add_argument(
        "--output_dir",
        type=str,
        default="output/vlm_output",
        help="Directory to save model outputs",
    )
    general_parser.add_argument(
        "--wandb_project",
        type=str,
        default="debug",
        help="WandB project name",
    )
    general_parser.add_argument(
        "--vision_input_path",
        type=str,
        required=True,
        help="Path to the vision input (either embeddings or images)",
    )
    general_parser.add_argument(
        "--embedding_constant",
        default=None,
        help="Constant value for training language only baseline (! Do not set this when training a VLM !). Can either be a float to set the whole embedding to a constant value or 'random' to set the embedding to a random values.",
    )
    general_parser.add_argument(
        "--train_only_vision_adapter",
        action="store_true",
        help="Train only the vision adapter (Qwen only)",
    )
    general_parser.add_argument(
        "--train_vision_encoder",
        action="store_true",
        help="Train the vision encoder (then embedding paths are the paths to the actual images).",
    )
    general_parser.add_argument(
        "--vision_encoder_path", type=str, 
        help="Path to vision encoder checkpoint. If None, use precomputed embeddings as input.",
        default=None
    )
    general_parser.add_argument(
        "--train_config", type=str, help="Path to config file (YAML or JSON)"
    )
    general_parser.add_argument(
        "--split_file", type=str, help="Path to cross validation split file (JSON)"
    )
    general_parser.add_argument(
        "--split_idx", type=int, default=None, help="Index of the split to use (0-4)"
    )
    general_parser.add_argument(
        "--dataset_name", type=str, default="ctrate", help="Name of the dataset"
    )
    general_parser.add_argument(
        "--n_train_samples", type=int, default=None, help="Number of training samples to use"
    )
    general_parser.add_argument(
        "--n_val_samples", type=int, default=50, help="Number of validation samples to use"
    )
    general_parser.add_argument(
        "--reduce_block",
        type=lambda s: [int(x) for x in s.strip("()").split(",")],
        default=[2, 2, 2],
        help="List for reduce_block, e.g. (2,2,2) or (4,4,4)",
    )
    general_parser.add_argument(
        "--do_resize",
        action="store_true",
        help="Resize images to 192x192x192 cubes (use this in combination with lung crops)",
    )
    general_parser.add_argument(
        "--do_transforms",
        action="store_true",
        help="If true, we do nnUNetv2 image augmentation.",
    )
    general_parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="If true, only use local files for qwen from_pretrained method.",
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

    if general_args.embedding_constant is not None:
        try:
            general_args.embedding_constant = float(general_args.embedding_constant)
        except ValueError:
            pass

    return general_args, sft_trainer_parsed


def ensure_wandb_config_logging():
    """
    Ensure that wandb properly logs hyperparameters in offline mode.
    This function updates the wandb config with environment variables and additional metadata.
    """
    if wandb.run is not None:
        # Add environment variables that are relevant to the training
        env_config = {
            "WANDB_MODE": os.getenv("WANDB_MODE", "online"),
            "WANDB_DIR": os.getenv("WANDB_DIR", ""),
            "SLURM_JOB_ID": os.getenv("SLURM_JOB_ID", ""),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "NODE_COUNT": os.getenv("SLURM_NNODES", "1"),
            "GPU_COUNT": os.getenv("GPUS_PER_NODE", "1"),
        }

        # Update wandb config with environment info
        wandb.config.update({"environment": env_config}, allow_val_change=True)

        # In offline mode, explicitly save the current config to ensure persistence
        if os.getenv("WANDB_MODE", "online").lower() == "offline":
            try:
                # Force config to be saved by accessing and logging all current config values
                config_dict = dict(wandb.config)

                # Log a summary of the configuration
                logger.info(
                    f"Offline mode - ensuring config persistence. Total config keys: {len(config_dict)}"
                )

                # Re-update the config to ensure it's properly saved in offline mode
                wandb.config.update(config_dict, allow_val_change=True)

                # Save config to a backup JSON file in the run directory for extra safety
                if hasattr(wandb.run, "dir") and wandb.run.dir:
                    config_backup_path = os.path.join(wandb.run.dir, "config_backup.json")
                    try:
                        import json

                        with open(config_backup_path, "w") as f:
                            json.dump(config_dict, f, indent=2, default=str)
                        logger.info(f"Config backup saved to: {config_backup_path}")
                    except Exception as e:
                        logger.warning(f"Could not save config backup: {e}")

            except Exception as e:
                logger.warning(f"Error in offline config handling: {e}")

        # Log the configuration to ensure it's saved
        logger.info(f"Updated wandb config with environment variables: {env_config}")
        logger.info(f"Total config keys logged: {len(dict(wandb.config))}")


def create_comprehensive_wandb_config(
    vlm_model, sft_trainer_config, general_args=None, dataset_info=None, world_size=1, rank=0
):
    """
    Create a comprehensive configuration dictionary for WandB logging.
    This ensures all hyperparameters are properly captured, especially for offline mode.
    """
    slurm_job_id = os.getenv("SLURM_JOB_ID", "local_run")
    username = os.getenv("USER", "unknown_user")

    # Start with basic model and training info
    config_dict = {
        "model_name": type(vlm_model).__name__,
        "model_id": getattr(vlm_model, "model_id", "unknown"),
        "trainable_params": vlm_model.get_trainable_params(),
        "training_args": sft_trainer_config,
        "slurm_job_id": slurm_job_id,
        "username": username,
        "timestamp": datetime.now().isoformat(),
        "wandb_mode": os.getenv("WANDB_MODE", "online"),
        "distributed_training": {
            "world_size": world_size,
            "rank": rank,
            "is_distributed": world_size > 1,
        }
    }

    # Add general args if available
    if general_args is not None:
        config_dict.update({"general_args": general_args})

    # Add dataset info if available
    if dataset_info is not None:
        config_dict.update({"dataset_info": dataset_info})

    # Add model-specific config if available
    if hasattr(vlm_model, "get_config_dict"):
        model_config = vlm_model.get_config_dict()
        config_dict.update({"model_config": model_config})

    # Add environment information
    env_config = {
        "WANDB_MODE": os.getenv("WANDB_MODE", "online"),
        "WANDB_DIR": os.getenv("WANDB_DIR", ""),
        "SLURM_JOB_ID": os.getenv("SLURM_JOB_ID", ""),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "NODE_COUNT": os.getenv("SLURM_NNODES", "1"),
        "GPU_COUNT": os.getenv("GPUS_PER_NODE", "1"),
        "PYTHONPATH": os.getenv("PYTHONPATH", ""),
        "TRANSFORMERS_OFFLINE": os.getenv("TRANSFORMERS_OFFLINE", ""),
        "HF_DATASETS_OFFLINE": os.getenv("HF_DATASETS_OFFLINE", ""),
        "MASTER_ADDR": os.getenv("MASTER_ADDR", ""),
        "MASTER_PORT": os.getenv("MASTER_PORT", ""),
        "WORLD_SIZE": os.getenv("WORLD_SIZE", "1"),
        "RANK": os.getenv("RANK", "0"),
        "LOCAL_RANK": os.getenv("LOCAL_RANK", "0"),
    }
    config_dict.update({"environment": env_config})

    return config_dict


def main():
    default_sft_trainer_config = {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "eval_steps": 250,
        "eval_strategy": "steps",
        # "save_strategy": "best",
        "save_strategy": "steps",
        "save_steps": 250,
        "save_total_limit": 2,
        "greater_is_better": True,
        "load_best_model_at_end": True,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "report_to": ["wandb"],
        # Memory management settings
        "torch_empty_cache_steps": 10,  # Clear cache every 10 steps
        "metric_for_best_model": "eval_loss", # changed to loss because other metrics are computed to rarely TODO fix this problem.. (quick fix beucase challenge deadline)
        "greater_is_better": False,
        # Distributed training specific settings
        "ddp_find_unused_parameters": False,
        "ddp_bucket_cap_mb": 25,
    }   
    general_args, sft_trainer_args = parse_args(default_sft_trainer_config)
    sft_trainer_config = vars(sft_trainer_args)

    # Log file setup (only on main process)
    if is_main_process():
        log_file = os.path.join(
            general_args.output_dir, datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_train_{uuid4()}.log")
        )
        logger.add(log_file)

        output_dir = os.path.join(
            general_args.output_dir, datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_train_{uuid4()}")
        )
        os.makedirs(output_dir, exist_ok=True)
        general_args.output_dir = output_dir

    if general_args.model_id is None:
        if general_args.model_type == "owl":
            general_args.model_id = "mPLUG/mPLUG-Owl3-1B-241014"
        elif general_args.model_type == "qwen":
            general_args.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Create the appropriate VLM model based on model_type
    if general_args.model_type == "owl":
        vlm_model_class = OwlVLModel
    elif general_args.model_type == "qwen":
        vlm_model_class = QwenVLModel
    else:
        raise ValueError(f"Unknown model type: {general_args.model_type}")
    # TODO: could add it as an arguement
    seed_everything(42)  # Set a random seed for reproducibility

    vlm_model = vlm_model_class(
        model_id=general_args.model_id,
        output_dir=general_args.output_dir,
        train_only_vision_adapter=general_args.train_only_vision_adapter,
        reduce_block=general_args.reduce_block,
        train_vision_encoder=general_args.train_vision_encoder,
        vision_encoder_path=general_args.vision_encoder_path,
        local_files_only=general_args.local_files_only
    )

    vision_input_type = VisionInputType.IMAGE if general_args.vision_encoder_path else VisionInputType.EMBEDDING
    # Create datasets
    if general_args.dataset_name == "amosmm":
        # Use AMOSMM dataset
        train_dataset = AMOSMMDataset(
            dataset_json_path=general_args.train_dataset_json_path,
            model_name=general_args.model_type,
            split="train",
            vision_input_path=general_args.vision_input_path,
            vision_input_type=vision_input_type,
            n_samples=general_args.n_train_samples,
            do_resize=general_args.do_resize,
            do_transforms=general_args.do_transforms,
            embedding_constant=general_args.embedding_constant,
        )

        val_dataset = AMOSMMDataset(
            dataset_json_path=general_args.val_dataset_json_path,
            model_name=general_args.model_type,
            split="val",
            vision_input_path=general_args.vision_input_path,
            vision_input_type=vision_input_type,
            n_samples=general_args.n_val_samples,  # Limit the validation set size
            do_resize=general_args.do_resize,
            do_transforms=general_args.do_transforms,
            embedding_constant=general_args.embedding_constant,
        )
    elif general_args.dataset_name == "ctrate":
        # Use CTRATE dataset
        train_dataset = CTRATEDataset(
            dataset_json_path=general_args.train_dataset_json_path,
            model_name=general_args.model_type,
            split="train" if general_args.split_idx is None else f"train_{general_args.split_idx}",
            vision_input_path=general_args.vision_input_path,
            vision_input_type=vision_input_type,
            n_samples=general_args.n_train_samples,
            split_files=general_args.split_file,
            answer_types=("TEXT",),
            do_resize=general_args.do_resize,
            do_transforms=general_args.do_transforms,
            embedding_constant=general_args.embedding_constant,
        )

        val_dataset = CTRATEDataset(
            dataset_json_path=general_args.val_dataset_json_path,
            model_name=general_args.model_type,
            split="val" if general_args.split_idx is None else f"val_{general_args.split_idx}",
            vision_input_path=general_args.vision_input_path,
            vision_input_type=vision_input_type,
            n_samples=general_args.n_val_samples,  # Limit the validation set size
            split_files=general_args.split_file,
            answer_types=("TEXT",),
            do_resize=general_args.do_resize,
            embedding_constant=general_args.embedding_constant,
            # no transforms for val dataset
        )

    if general_args.embedding_constant is not None and is_main_process():
        logger.info(f"Using embedding constant: {general_args.embedding_constant}")
        print(f"Using embedding constant: {general_args.embedding_constant}")

    if is_main_process():
        logger.info(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
        )
        print(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
        )
        logger.info(f"Using train batch size: {sft_trainer_config['per_device_train_batch_size']}")
        logger.info(f"Using eval batch size: {sft_trainer_config['per_device_eval_batch_size']}")

    # Create the trainer and train the model
    trainer = UnifiedVLMTrainer(
        vlm_model,
        output_dir=general_args.output_dir,
        sft_trainer_config=sft_trainer_config,
    )

    # Update trainer with additional config for wandb logging
    trainer.general_args = vars(general_args)
    trainer.dataset_info = {
        "dataset_name": general_args.dataset_name,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }

    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
