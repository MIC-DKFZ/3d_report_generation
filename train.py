import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import wandb
from configs.run_config import RunConfig
from data.datasets import AMOSMMDataset, CTRATEDataset
from models.abstract_vlm import AbstractVLMModel
from owl.callbacks import LossLoggingCallback, PrintEvalBatchCallback
from registries.vlm_registry import vlm_registry
from utils import logger
from utils.helper import (
    is_main_process,
    load_config_file,
    merge_config_with_cli_args,
    print_model_parameters,
    seed_everything,
)
from utils.load_and_save_files import load_json, save_json, save_yaml
from utils.memory_utils import (
    aggressive_cleanup,
    log_memory_usage,
    setup_memory_efficient_training,
)
from utils.paths import VLM_CONFIG_DIR, VLM_OUTPUT_DIR
from utils.vision_input_type import VisionInputType


def setup_distributed_training():
    """
    Set up distributed training environment.
    This function handles the distributed training initialization as recommended in the JSC tutorial.
    """
    # Check if we're in a distributed environment
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE"))

        logger.info(f"Distributed training detected: rank={rank}, world_size={world_size}, "
                    f"local_rank={local_rank}, local_world_size={local_world_size}")

        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        # Initialize the process group
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            logger.info(f"Process group initialized on rank {rank}")
        
        return True, rank, world_size, local_rank, local_world_size
    else:
        logger.info("Single GPU/CPU training detected")
        return False, 0, 1, 0, 1


class UnifiedVLMTrainer:
    def __init__(
            self,
            vlm_model: AbstractVLMModel,
            cfg: RunConfig,
    ):
        """
        Initialize the unified VLM trainer.

        Args:
            vlm_model: An instance of a class derived from AbstractVLMModel
            cfg: The RunConfig of the experiment
        """
        self.vlm_model: AbstractVLMModel = vlm_model
        self.trainer: SFTTrainer | None = None
        self.cfg: RunConfig = cfg

        # Setup distributed training
        (self.is_distributed,
         self.rank,
         self.world_size,
         self.local_rank,
         self.local_world_size) = setup_distributed_training()

    def get_training_args(self):
        """Configure standard training arguments for distributed training"""
        num_cores = os.cpu_count()
        
        # Adjust number of workers based on distributed setup
        if self.is_distributed:
            # Reduce workers per process in distributed training
            num_workers = max(1, (num_cores // 4) // self.local_world_size)
        else:
            num_workers = 2

        # TODO: Do calculation before final_config.yml is logged
        self.cfg.trainer.dataloader_num_workers = num_workers

        # Add model-specific arguments
        model_args = self.vlm_model.get_model_specific_training_args()

        return SFTConfig(**self.cfg.trainer.model_dump(),
                         **model_args,
                         run_name=self.cfg.general.run_name,
                         output_dir=self.cfg.general.output_dir)

    def get_quantization_config(self):
        """Create and return the BitsAndBytes quantization config"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

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


        # Create quantization config if needed
        bnb_config = None
        # Uncomment if you want to use quantization
        # if trainable_params.get("train_only_vision_adapter", False) == True:
        #     bnb_config = self.get_quantization_config()
        # else:
        #     bnb_config = None

        # Setup base model with quantization config if applicable
        model = self.vlm_model.setup_model(bnb_config=bnb_config)

        # Apply trainable parameter configs and possibly add PEFT adapter layers
        self.vlm_model.apply_trainable_config()

        # Log trainable params including their 'requires_grad' status
        if is_main_process():
            trainable_params_log_path = Path(self.cfg.general.output_dir, "model_parameters.log")

            with open(trainable_params_log_path, "w") as f:
                # With 'remove_duplicate=True', tied/shared parameters like 'lm_head.weight'
                # are only listed once in named_parameters(), under their first occurrence.
                print_model_parameters(model=model, file=f, remove_duplicate=True)

        # Create comprehensive config for WandB logging
        config_dict = create_comprehensive_wandb_config(
            self.vlm_model, self.cfg,
            self.world_size, self.rank
        )

        # Only initialize wandb on the main process (rank 0) for DDP
        if is_main_process():
            wandb.init(
                entity="mic-rocket-vlm",
                project=self.cfg.general.wandb_project,
                name=self.cfg.general.run_name,
                reinit=True,
                group=f"{type(self.vlm_model).__name__}_run",
                config=config_dict,
                id=str(Path(self.cfg.general.output_dir).relative_to(VLM_OUTPUT_DIR)).replace("/", "--")
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
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.vlm_model.collate_fn,
            callbacks=callbacks
        )

        # Train model
        if is_main_process():
            log_memory_usage("Before Training")
        
        self.trainer.train()
        
        if is_main_process():
            log_memory_usage("After Training")

        # Save model (only on main process)
        # if is_main_process():
        #     self.trainer.save_model(self.cfg.general.output_dir)

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


def setup_config_and_logging() -> RunConfig:
    args_parser = argparse.ArgumentParser(description="Train a Vision Language Model")

    args_parser.add_argument(
        "--train_config", type=str, required=True,
        help="Path to config file (YAML or JSON), relative to the 'VLM_CONFIG_DIR' environment variable"
    )

    args_parser.add_argument(
        "--add_base_configs",
        type=str,
        nargs="+",
        default=[],
        help="Optional list of base config files"
    )

    args, _ = args_parser.parse_known_args()

    train_config_path = Path(VLM_CONFIG_DIR, args.train_config)
    if not train_config_path.exists():
        raise ValueError(f"The provided config file does not exist: {train_config_path}")

    cfg_dict = load_config_file(train_config_path)

    if args.add_base_configs:
        # Give base configs that are passed via CLI higher prio
        cfg_dict["base_configs"] = args.add_base_configs + cfg_dict.get("base_configs", [])

    # Merge base configs into cfg_dict
    cfg_dict = RunConfig.apply_base_configs(cfg_dict=cfg_dict, config_dir=VLM_CONFIG_DIR)

    # Parse all args into their respective Pydantic models/configs
    cfg = RunConfig.model_validate(cfg_dict)

    cfg.general.train_config = str(train_config_path)
    cfg.resolve_run_name()
    cfg.resolve_output_dir()

    # Setup output directory and log file
    if is_main_process():
        os.makedirs(cfg.general.output_dir, exist_ok=True)
        log_file_path = os.path.join(cfg.general.output_dir, "train.log")
        logger.add(log_file_path)
        logger.info(f"Successfully set up output directory at {cfg.general.output_dir}")
        print(f"Successfully set up output directory at {cfg.general.output_dir}")

        shutil.copyfile(cfg.general.train_config, os.path.join(cfg.general.output_dir, "input_config.yml"))

    # Reset some args in case the vision input are embedding (instead of images)
    cfg.maybe_reset_vision_encoder_specific_args(logger=logger)

    # Apply CLI argument overrides to configs (intended for testing/debugging only)
    cfg.general = merge_config_with_cli_args(cfg.general, ignore_keys="train_config", parent_key="general")
    cfg.model = merge_config_with_cli_args(cfg.model, parent_key="model")
    cfg.trainer = merge_config_with_cli_args(cfg.trainer, parent_key="trainer")
    cfg.data = merge_config_with_cli_args(cfg.data, parent_key="data")

    # Resolve model_id if not provided
    if cfg.model.model_id is None and not cfg.model.model_type in vlm_registry:
        raise ValueError(f"Unknown model type: {cfg.model.model_type}")
    cfg.model.model_id = cfg.model.model_id or vlm_registry[cfg.model.model_type].model_id

    # Log the final configuration
    if is_main_process():
        final_config_path = os.path.join(cfg.general.output_dir, "final_config.yml")
        save_yaml(
            {
                "command": " ".join(sys.argv),
                "run": cfg.model_dump()
            },
            final_config_path,
            sort_keys=False
        )

    return cfg


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
                        save_json(config_dict, config_backup_path, default=str)
                        logger.info(f"Config backup saved to: {config_backup_path}")
                    except Exception as e:
                        logger.warning(f"Could not save config backup: {e}")

            except Exception as e:
                logger.warning(f"Error in offline config handling: {e}")

        # Log the configuration to ensure it's saved
        logger.info(f"Updated wandb config with environment variables: {env_config}")
        logger.info(f"Total config keys logged: {len(dict(wandb.config))}")


def create_comprehensive_wandb_config(
        vlm_model: AbstractVLMModel,
        cfg: RunConfig,
        world_size: int = 1,
        rank: int = 0
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
        "run_config": cfg.model_dump(),
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

@logger.catch(reraise=True)
def main():
    cfg = setup_config_and_logging()

    # TODO: could add it as an argument
    seed_everything(42)  # Set a random seed for reproducibility

    try:
        vlm_model_class = vlm_registry[cfg.model.model_type].model_class
    except KeyError:
        raise ValueError(f"Unknown model type: {cfg.general.model_type}")

    vlm_model = vlm_model_class(
        model_cfg=cfg.model,
        vision_input_type=cfg.data.vision_input_type,
        local_files_only=cfg.general.local_files_only
    )

    if cfg.data.vision_input_type == VisionInputType.IMAGE:
        ve_plans = load_json(Path(cfg.model.vision_encoder.vision_encoder_path, "plans.json"))
        vision_encoder_patch_size = ve_plans["configurations"]["3d_fullres"]["patch_size"]
    else:
        vision_encoder_patch_size = None

    # Create datasets
    if cfg.data.dataset_name == "amosmm":
        # Use AMOSMM dataset
        train_dataset = AMOSMMDataset(
            dataset_json_path=cfg.data.train_dataset_json_path,
            model_name=cfg.model.model_type,
            split="train",
            vision_input_path=cfg.data.vision_input_path,
            vision_input_type=cfg.data.vision_input_type,
            answer_types=cfg.data.answer_types,
            n_samples=cfg.data.n_train_samples,
            do_resize=cfg.data.do_resize,
            do_transforms=cfg.data.do_transforms,
            embedding_constant=cfg.data.embedding_constant,
            vision_encoder_patch_size=vision_encoder_patch_size
        )

        val_dataset = AMOSMMDataset(
            dataset_json_path=cfg.data.val_dataset_json_path,
            model_name=cfg.model.model_type,
            split="val",
            vision_input_path=cfg.data.vision_input_path,
            vision_input_type=cfg.data.vision_input_type,
            n_samples=cfg.data.n_val_samples,  # Limit the validation set size
            answer_types=cfg.data.answer_types,
            do_resize=cfg.data.do_resize,
            do_transforms=cfg.data.do_transforms,
            embedding_constant=cfg.data.embedding_constant,
            vision_encoder_patch_size=vision_encoder_patch_size,
        )
    elif cfg.data.dataset_name == "ctrate":
        # Use CTRATE dataset
        train_dataset = CTRATEDataset(
            dataset_json_path=cfg.data.train_dataset_json_path,
            model_name=cfg.model.model_type,
            split="train",
            vision_input_path=cfg.data.vision_input_path,
            vision_input_type=cfg.data.vision_input_type,
            n_samples=cfg.data.n_train_samples,
            answer_types=cfg.data.answer_types,
            do_resize=cfg.data.do_resize,
            do_transforms=cfg.data.do_transforms,
            embedding_constant=cfg.data.embedding_constant,
            vision_encoder_patch_size=vision_encoder_patch_size
        )

        val_dataset = CTRATEDataset(
            dataset_json_path=cfg.data.val_dataset_json_path,
            model_name=cfg.model.model_type,
            split="val",
            vision_input_path=cfg.data.vision_input_path,
            vision_input_type=cfg.data.vision_input_type,
            n_samples=cfg.data.n_val_samples,  # Limit the validation set size
            answer_types=cfg.data.answer_types,
            do_resize=cfg.data.do_resize,
            embedding_constant=cfg.data.embedding_constant,
            # no transforms for val dataset
            vision_encoder_patch_size=vision_encoder_patch_size
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.data.dataset_name}")

    if cfg.data.embedding_constant is not None and is_main_process():
        logger.info(f"Using embedding constant: {cfg.data.embedding_constant}")
        print(f"Using embedding constant: {cfg.data.embedding_constant}")

    if is_main_process():
        logger.info(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
        )
        print(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
        )
        logger.info(f"Using train batch size: {cfg.trainer.per_device_train_batch_size}")
        logger.info(f"Using eval batch size: {cfg.trainer.per_device_eval_batch_size}")

    # Create the trainer and train the model
    trainer = UnifiedVLMTrainer(
        vlm_model=vlm_model,
        cfg=cfg
    )

    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
