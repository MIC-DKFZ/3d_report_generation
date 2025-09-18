import argparse
import os
import shutil
import sys
from datetime import datetime

import torch
import torch.distributed as dist
import yaml
from peft import get_peft_model
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import wandb
from data.datasets import AMOSMMDataset, CTRATEDataset
from models.abstract_vlm import AbstractVLMModel
from owl.callbacks import LossLoggingCallback, PrintEvalBatchCallback
from registry import vlm_registry
from utils import logger
from utils.config_models import GeneralConfig, SFTTrainerConfig, CoreConfig, RunConfig
from utils.helper import seed_everything, load_config_file, merge_config_with_cli_args
from utils.memory_utils import (
    aggressive_cleanup,
    log_memory_usage,
    setup_memory_efficient_training,
)


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


def is_main_process():
    """
    Check if this is the main process (rank 0).
    As recommended in the JSC tutorial, only rank 0 should handle logging, saving, etc.
    """
    return int(os.environ.get("RANK", "0")) == 0


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

        # Setup output directory (only on main process)
        if is_main_process():
            self.setup_logging_dir()

    def get_training_args(self):
        """Configure standard training arguments for distributed training"""
        num_cores = os.cpu_count()
        
        # Adjust number of workers based on distributed setup
        if self.is_distributed:
            # Reduce workers per process in distributed training
            num_workers = max(1, (num_cores // 4) // self.local_world_size)
        else:
            num_workers = 2

        self.cfg.trainer.dataloader_num_workers = num_workers

        # Add model-specific arguments
        model_args = self.vlm_model.get_model_specific_training_args()

        return SFTConfig(**self.cfg.trainer.model_dump(), **model_args)

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

    def setup_logging_dir(self) -> None:
        """
        Prepare the logging directory and initialize its contents:
            - input_config.yml: a copy of the provided input config file
            - final_config.yml: final merged configuration (defaults + input config + CLI args)
            - train.log: log file for runtime logs
        """
        cfg = self.cfg

        os.makedirs(cfg.general.output_dir, exist_ok=True)
        shutil.copyfile(cfg.core.train_config, os.path.join(cfg.general.output_dir, "input_config.yml"))
        log_file_path = os.path.join(cfg.general.output_dir, "train.log")
        logger.add(log_file_path)

        # Save final run config
        final_config_path = os.path.join(cfg.general.output_dir, "final_config.yml")
        with open(final_config_path, "w") as f:
            yaml.safe_dump({
                "command": " ".join(sys.argv),
                "run": cfg.model_dump()
            }, f)

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
        config_dict = create_comprehensive_wandb_config(    # TODO: DataConfig
            self.vlm_model, self.cfg,
            self.world_size, self.rank
        )
        # Ensure trainable_params are included in the config
        config_dict["trainable_params"] = trainable_params  # TODO: somehow add this to RunConfig

        # Only initialize wandb on the main process (rank 0) for DDP
        if is_main_process():
            wandb.init(
                entity="mic-rocket-vlm",
                project=self.cfg.general.wandb_project,
                name=datetime.now().strftime(
                    f"{self.cfg.general.run_name}_%Y-%m-%d_%H-%M-%S"
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
            self.trainer.save_model(self.cfg.general.output_dir)

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


def parse_args() -> RunConfig:
    """
    Returns:
        RunConfig: Currently, three configuration are held by the returned RunConfig object:
            - core (CoreConfig): (dataset and config file)
            - general (GeneralConfig): (model configs, data augs, num train samples, ...)
            - trainer (SFTTrainerConfig): (epochs, eval & save steps, hyperparams, ...)
    """

    core_args_parser = argparse.ArgumentParser(description="Train a Vision Language Model")

    core_args_parser.add_argument(
        "--train_config", type=str, required=True,
        help="Path to config file (YAML or JSON), relative to the 'VLM_CONFIG_DIR' environment variable"
    )
    core_args_parser.add_argument(
        "--dataset_name", type=str, required=True, choices=["ctrate", "amosmm"], help="Name of the dataset"
    )
    core_args_parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset JSON file"
    )
    core_args_parser.add_argument(
        "--embeddings_path", type=str, required=True, help="Path to the vision embeddings",
    )

    core_args, _ = core_args_parser.parse_known_args()

    core_args.train_config = os.path.join(os.getenv("VLM_CONFIG_DIR"), core_args.train_config)
    if not os.path.exists(core_args.train_config):
        raise ValueError(f"The provided config file does not exist: {core_args.train_config}")

    config_file = load_config_file(core_args.train_config)
    general_args_config_file, sft_trainer_args_config_file = config_file["general"], config_file["trainer"]

    # Resolve model_id if not provided
    if general_args_config_file.get("model_id") is None and general_args_config_file["model_type"] not in vlm_registry:
        raise ValueError(f"Unknown model type: {general_args_config_file['model_type']}")
    model_id = general_args_config_file.get("model_id") or vlm_registry[general_args_config_file["model_type"]].model_id

    # Parse all args into their respective Pydantic models/configs
    cfg = RunConfig(
        core=CoreConfig(**vars(core_args)),
        general=GeneralConfig(**general_args_config_file, model_id=model_id),
        trainer=SFTTrainerConfig(**sft_trainer_args_config_file)
    )

    # Apply CLI argument overrides to configs (intended for testing/debugging only)
    cfg.general = merge_config_with_cli_args(cfg.general)
    cfg.trainer = merge_config_with_cli_args(cfg.trainer)

    # Use the config filename (without extension) as both 'run_name' and the results folder name later on
    config_name, _ = os.path.splitext(os.path.basename(cfg.core.train_config))
    cfg.general.run_name = datetime.now().strftime(f"{config_name}_%Y-%m-%d_%H-%M-%S")

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
        "trainable_params": vlm_model.get_trainable_params(),
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


def main():

    cfg = parse_args()

    vlm_config_dir, vlm_output_dir = os.getenv("VLM_CONFIG_DIR"), os.getenv("VLM_OUTPUT_DIR")
    cfg.general.output_dir = os.path.join(
        vlm_output_dir,
        cfg.core.dataset_name,
        os.path.basename(cfg.core.embeddings_path),
        os.path.dirname(os.path.relpath(cfg.core.train_config, vlm_config_dir)),
        cfg.general.run_name
    )

    # TODO: could add it as an argument
    seed_everything(42)  # Set a random seed for reproducibility

    try:
        vlm_model_class = vlm_registry[cfg.general.model_type].model_class
    except KeyError:
        raise ValueError(f"Unknown model type: {cfg.general.model_type}")

    vlm_model = vlm_model_class(
        model_id=cfg.general.model_id,
        output_dir=cfg.general.output_dir,
        train_only_vision_adapter=cfg.general.train_only_vision_adapter,
        reduce_block=cfg.general.reduce_block,
        train_vision_encoder=cfg.general.train_vision_encoder,
        vision_encoder_path=cfg.general.vision_encoder_path,
        freeze_vision_encoder=cfg.general.freeze_vision_encoder,
        local_files_only=cfg.general.local_files_only
    )

    # Create datasets
    if cfg.core.dataset_name == "amosmm":
        # Use AMOSMM dataset
        train_dataset = AMOSMMDataset(
            dataset_json_path=cfg.core.dataset_path,
            model_name=cfg.general.model_type,
            split="train",
            embeddings_path=cfg.core.embeddings_path,
            cache_file=cfg.general.dataset_cache,  # Use cache file if provided
            embedding_constant=cfg.general.embedding_constant,
        )

        val_dataset = AMOSMMDataset(
            dataset_json_path=cfg.core.dataset_path,
            model_name=cfg.general.model_type,
            split="val",
            embeddings_path=cfg.core.embeddings_path,
            n_samples=cfg.general.n_val_samples,  # Limit the validation set size
            cache_file=cfg.general.dataset_cache_val,  # Use cache file if provided
            embedding_constant=cfg.general.embedding_constant,
        )
    elif cfg.core.dataset_name == "ctrate":
        # Use CTRATE dataset
        train_dataset = CTRATEDataset(
            dataset_json_path=cfg.core.dataset_path,
            model_name=cfg.general.model_type,
            split="train" if cfg.general.split_idx is None else f"train_{cfg.general.split_idx}",
            vqa_path=cfg.general.vqa_path,
            embeddings_path=cfg.core.embeddings_path,
            n_samples=cfg.general.n_train_samples,
            split_files=cfg.general.split_file,
            answer_types=("TEXT",),
            cache_file=cfg.general.dataset_cache,  # Use cache file if provided
            load_image=cfg.general.train_vision_encoder,
            do_resize=cfg.general.do_resize,
            do_transforms=cfg.general.do_transforms,
            embedding_constant=cfg.general.embedding_constant,
            use_csv_reports=cfg.general.use_csv_reports
        )

        val_dataset = CTRATEDataset(
            dataset_json_path=cfg.core.dataset_path,
            model_name=cfg.general.model_type,
            split="val" if cfg.general.split_idx is None else f"val_{cfg.general.split_idx}",
            embeddings_path=cfg.core.embeddings_path,
            n_samples=cfg.general.n_val_samples,  # Limit the validation set size
            split_files=cfg.general.split_file,
            answer_types=("TEXT",),
            cache_file=cfg.general.dataset_cache_val,  # Use cache file if provided
            load_image=cfg.general.train_vision_encoder,
            do_resize=cfg.general.do_resize,
            embedding_constant=cfg.general.embedding_constant,
            use_csv_reports=cfg.general.use_csv_reports
            # no transforms for val dataset
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.core.dataset_name}")

    if cfg.general.embedding_constant is not None and is_main_process():
        logger.info(f"Using embedding constant: {cfg.general.embedding_constant}")
        print(f"Using embedding constant: {cfg.general.embedding_constant}")

    if is_main_process():
        logger.info(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
        )
        print(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
        )
        logger.info(f"Using batch size: {cfg.trainer.per_device_train_batch_size}")

    # Create the trainer and train the model
    trainer = UnifiedVLMTrainer(
        vlm_model=vlm_model,
        cfg=cfg
    )

    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
