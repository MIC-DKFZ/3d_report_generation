"""W&B run setup and configuration logging for training."""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

import wandb

from vlm.utils import logger
from vlm.utils.ddp import is_main_process
from vlm.utils.serialization import save_json

if TYPE_CHECKING:
    from vlm.configs.run_config import RunConfig
    from vlm.models.abstract_vl_wrapper import AbstractVLWrapper


WANDB_CONFIG_BACKUP_FILENAME = "config_backup.json"


def initialize_wandb(
    vlm_model: AbstractVLWrapper, cfg: RunConfig, world_size: int = 1, rank: int = 0
) -> None:
    """Initialize the main process's run and persist its configuration."""
    # Create comprehensive config for WandB logging
    config_dict = create_comprehensive_wandb_config(vlm_model, cfg, world_size, rank)

    # Only initialize wandb on the main process (rank 0) for DDP
    if is_main_process():
        wandb.init(
            entity=cfg.general.wandb_entity,
            project=cfg.general.wandb_project,
            name=cfg.general.run_name,
            reinit=True,
            group=f"{type(vlm_model).__name__}_run",
            config=config_dict,
            id=cfg.general.run_name,
        )

        # Ensure wandb properly logs hyperparameters in offline mode
        ensure_wandb_config_logging()

        # Additional explicit config update for offline mode robustness
        if os.getenv("WANDB_MODE", "online").lower() == "offline":
            # Re-update config one more time to ensure persistence
            wandb.config.update(config_dict, allow_val_change=True)
            logger.info_main("Offline mode: Config explicitly updated after init")


def finish_wandb() -> None:
    """Verify offline configuration and finish the main process's run."""
    # Final config validation for offline mode
    if os.getenv("WANDB_MODE", "online").lower() == "offline" and wandb.run is not None:
        final_config = dict(wandb.config)
        logger.info_main(f"Final config verification - Total keys: {len(final_config)}")
        if len(final_config) < 5:  # Arbitrary threshold - expect at least some config
            logger.warning_main(
                "Suspiciously few config keys logged - may indicate config persistence issue"
            )

    if is_main_process():
        wandb.finish()


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
                logger.info_main(
                    f"Offline mode - ensuring config persistence. Total config keys: {len(config_dict)}"
                )

                # Re-update the config to ensure it's properly saved in offline mode
                wandb.config.update(config_dict, allow_val_change=True)

                # Save config to a backup JSON file in the run directory for extra safety
                if hasattr(wandb.run, "dir") and wandb.run.dir:
                    config_backup_path = os.path.join(
                        wandb.run.dir, WANDB_CONFIG_BACKUP_FILENAME
                    )
                    try:
                        save_json(config_dict, config_backup_path, default=str)
                        logger.info_main(f"Config backup saved to: {config_backup_path}")
                    except Exception as e:
                        logger.warning_main(f"Could not save config backup: {e}")

            except Exception as e:
                logger.warning_main(f"Error in offline config handling: {e}")

        # Log the configuration to ensure it's saved
        logger.info_main(f"Updated wandb config with environment variables: {env_config}")
        logger.info_main(f"Total config keys logged: {len(dict(wandb.config))}")


def create_comprehensive_wandb_config(
    vlm_model: AbstractVLWrapper, cfg: RunConfig, world_size: int = 1, rank: int = 0
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
        },
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
