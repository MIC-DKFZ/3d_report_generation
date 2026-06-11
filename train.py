import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader
from trl import SFTConfig

import wandb
from vlm.configs.run_config import RunConfig
from vlm.data.datasets import (
        AMOSMMDataset,
        CTRATEDataset,
        MerlinDataset,
        MerlinDatasetRetrieval,
        MerlinDatasetRetrievalWithImage,
        CTRATEDatasetRetrieval,
        CTRATEDatasetRetrievalWithImage,
        ToothFairy4Dataset,
)
from vlm.evaluation.eval_distributed_sampler import EvalDistributedSampler
from inference import main as inference_main
from vlm.models.abstract_vlm import AbstractVLMModel
from vlm.training.callbacks.logging import LossLoggingCallback, PycocoevalCallback, WandbGroupedLRCallback, \
    TerminateOnNaNLossCallback
from vlm.training.trainers import VLMSFTTrainer
from vlm.registries.vlm_registry import VLM_REGISTRY
from vlm.utils import logger
from vlm.utils.ddp_utils import setup_ddp
from vlm.utils.helper import seed_everything, load_config_file, \
    is_main_process, count_parameters
from vlm.utils.load_and_save_files import save_json, save_yaml
from vlm.utils.logging import setup_logger, log_banner
from vlm.utils.memory_utils import (
    aggressive_cleanup,
    log_memory_usage,
    setup_memory_efficient_training,
)
from vlm.utils.paths import VLM_CONFIG_DIR


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
        self.trainer: VLMSFTTrainer | None = None
        self.cfg: RunConfig = cfg

        (
            self.is_distributed,
            self.rank,
            self.world_size,
            self.local_rank,
            self.local_world_size,
        ) = setup_ddp()

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

        # Normalize legacy placeholder values so non-DeepSpeed runs do not try
        # to parse "deepspeed" as a config/path.
        deepspeed_cfg_path = self.cfg.trainer.deepspeed
        if isinstance(deepspeed_cfg_path, str):
            normalized_ds = deepspeed_cfg_path.strip().lower()
            if normalized_ds in {"", "none", "null", "false", "off", "deepspeed"}:
                logger.warning_main(
                    "Ignoring placeholder value for trainer.deepspeed "
                    f"({deepspeed_cfg_path!r}) and disabling DeepSpeed."
                )
                self.cfg.trainer.deepspeed = None
                deepspeed_cfg_path = None

        # Keep HF/Accelerate and DeepSpeed gradient accumulation in sync to avoid
        # "Gradient accumulation steps mismatch" warnings.
        if deepspeed_cfg_path:
            ds_path = Path(deepspeed_cfg_path)
            if not ds_path.is_absolute():
                ds_path = Path.cwd() / ds_path
            if ds_path.exists():
                with open(ds_path, "r", encoding="utf-8") as f:
                    ds_cfg = json.load(f)
                ds_grad_accum = ds_cfg.get("gradient_accumulation_steps")
                if isinstance(ds_grad_accum, int) and ds_grad_accum >= 1:
                    if self.cfg.trainer.gradient_accumulation_steps != ds_grad_accum:
                        logger.warning_main(
                            "Overriding trainer.gradient_accumulation_steps "
                            f"({self.cfg.trainer.gradient_accumulation_steps}) with DeepSpeed value "
                            f"({ds_grad_accum}) from {ds_path}."
                        )
                    self.cfg.trainer.gradient_accumulation_steps = ds_grad_accum

        # Add model-specific arguments
        model_args = self.vlm_model.get_model_specific_training_args()

        return SFTConfig(
            **self.cfg.trainer.model_dump(),
            **model_args,
            run_name=self.cfg.general.run_name,
            output_dir=self.cfg.general.output_dir,
        )

    def train(self, train_dataset, val_dataset):
        """
        Main training function that trains the VLM model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        if self.is_distributed:
            dist.barrier()

        # Setup memory efficient training
        setup_memory_efficient_training()
        if is_main_process():
            log_memory_usage("Before Training Setup")

        # Setup model, processor and tokenizer
        if not self.cfg.general.continue_training:
            model = self.vlm_model.setup_model()
            # Apply trainable parameter configs and possibly add PEFT adapter layers
            self.vlm_model.apply_trainable_config()
        else:
            # The loaded model should already have the trainable parameters applied
            model = self.vlm_model.model

        if self.cfg.general.init_from_checkpoint_dir:
            missing_keys, unexpected_keys = self.vlm_model.load_weights_from_checkpoint_dir(
                self.cfg.general.init_from_checkpoint_dir,
                allow_unexpected_keys=True,  # Needed if Stage N+1 needs additional modules
            )
            preview_limit = 20

            model_for_key_check = self.vlm_model.model
            if model_for_key_check is None:
                logger.warning_main(
                    "Checkpoint load: could not classify missing keys because model is None"
                )
                trainable_param_names = set()
            else:
                trainable_param_names = {
                    name for name, p in model_for_key_check.named_parameters() if p.requires_grad
                }

            if missing_keys:
                missing_trainable = [k for k in missing_keys if k in trainable_param_names]
                missing_non_trainable = [k for k in missing_keys if k not in trainable_param_names]

                if missing_trainable:
                    missing_trainable_preview = missing_trainable[:preview_limit]
                    missing_trainable_suffix = "" if len(missing_trainable) <= preview_limit else " ..."
                    logger.warning_main(
                        f"Checkpoint load: {len(missing_trainable)} trainable keys are missing from "
                        f"{self.cfg.general.init_from_checkpoint_dir}. "
                        f"Preview (first {len(missing_trainable_preview)}): "
                        f"{missing_trainable_preview}{missing_trainable_suffix}"
                    )

                if missing_non_trainable:
                    missing_non_trainable_preview = missing_non_trainable[:preview_limit]
                    missing_non_trainable_suffix = (
                        "" if len(missing_non_trainable) <= preview_limit else " ..."
                    )
                    logger.info_main(
                        f"Checkpoint load: {len(missing_non_trainable)} non-trainable keys are missing from "
                        f"{self.cfg.general.init_from_checkpoint_dir} (usually expected for trainable-only checkpoints). "
                        f"Preview (first {len(missing_non_trainable_preview)}): "
                        f"{missing_non_trainable_preview}{missing_non_trainable_suffix}"
                    )

            if unexpected_keys:
                unexpected_preview = unexpected_keys[:preview_limit]
                unexpected_suffix = "" if len(unexpected_keys) <= preview_limit else " ..."
                logger.warning_main(
                    f"Checkpoint load: {len(unexpected_keys)} unexpected keys in "
                    f"{self.cfg.general.init_from_checkpoint_dir}. "
                    f"Preview (first {len(unexpected_preview)}): {unexpected_preview}{unexpected_suffix}"
                )
            if not missing_keys and not unexpected_keys:
                logger.warning_main(
                    "Checkpoint load: no missing/unexpected keys when initializing from "
                    f"{self.cfg.general.init_from_checkpoint_dir}"
                )
            # Keep trainer input explicit after checkpoint weight load
            model = self.vlm_model.model

        # Log adapter-specific numbers (e.g. number of params)
        if self.vlm_model.vision_adapter is not None:
            adapter_parameter_count = count_parameters(self.vlm_model.vision_adapter)
            _stats = {
                "parameter_count": adapter_parameter_count
            }
            save_json(_stats, Path(self.cfg.general.output_dir, "adapter_stats.json"))

        # Create comprehensive config for WandB logging
        config_dict = create_comprehensive_wandb_config(
            self.vlm_model, self.cfg, self.world_size, self.rank
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
                # id=f"{str(Path(self.cfg.data.vision_input_path).name)}--{self.cfg.general.run_name}"
                id=self.cfg.general.run_name
            )

            # Ensure wandb properly logs hyperparameters in offline mode
            ensure_wandb_config_logging()

            # Additional explicit config update for offline mode robustness
            if os.getenv("WANDB_MODE", "online").lower() == "offline":
                # Re-update config one more time to ensure persistence
                wandb.config.update(config_dict, allow_val_change=True)
                logger.info_main("Offline mode: Config explicitly updated after init")

        # Configure training arguments
        training_args = self.get_training_args()

        callbacks = [
            LossLoggingCallback(),
            WandbGroupedLRCallback(),
            TerminateOnNaNLossCallback(threshold=3)
        ]

        if val_dataset is not None:
            if self.is_distributed:
                sampler = EvalDistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False
                )
            else:
                sampler = None

            eval_dataloader = DataLoader(val_dataset, batch_size=self.cfg.trainer.per_device_eval_batch_size, shuffle=False,
                                         sampler=sampler,
                                         collate_fn=self.vlm_model.collate_fn,
                                         num_workers=self.cfg.eval.num_dataloader_workers)

            callbacks.append(
                # PrintEvalBatchCallback(  # type: ignore[arg-type]
                #     eval_dataloader=eval_dataloader,
                #     task_type="RG" if self.cfg.data.dataset_name == "ctrate" else "VQA",  # TODO: temporary hardcode
                #     max_new_tokens=self.cfg.eval.max_new_tokens,
                #     eval_every_n_epochs=self.cfg.eval.eval_every_n_epochs,
                #     tokenizer=self.vlm_model.tokenizer,
                #     processor=self.vlm_model.processor,
                #     is_distributed=self.is_distributed
                # )
                PycocoevalCallback(  # type: ignore[arg-type]
                    eval_dataloader=eval_dataloader,
                    max_new_tokens=self.cfg.eval.max_new_tokens,
                    eval_every_n_epochs=self.cfg.eval.eval_every_n_epochs,
                    tokenizer=self.vlm_model.tokenizer,
                    processor=self.vlm_model.processor,
                    is_distributed=self.is_distributed,
                    calculate_radbert_score=self.cfg.eval.calculate_radbert_score
                )
            )

        # Initialize SFT trainer
        init_opt_scheduler_ckpt = None
        if (
            self.cfg.general.init_optimizer_scheduler_from_checkpoint
            and self.cfg.general.init_from_checkpoint_dir
            and not self.cfg.general.continue_training
        ):
            init_opt_scheduler_ckpt = self.cfg.general.init_from_checkpoint_dir

        self.trainer = VLMSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.vlm_model.collate_fn,
            callbacks=callbacks,
            init_opt_scheduler_from_ckpt=init_opt_scheduler_ckpt,
            scheduler=self.cfg.scheduler.to_instance(
                module_attr_map=self.vlm_model.module_attr_map,
                forbidden_name_patterns_for_wd=self.vlm_model.forbidden_name_patterns_for_wd,
                output_dir=self.cfg.general.output_dir
            ),
        )

        if self.is_distributed:
            dist.barrier()

        # Train model
        if is_main_process():
            log_memory_usage("Before Training")

        if self.cfg.general.continue_training:
            if self.cfg.general.continue_training_mode == "final":
                self.trainer.train(resume_from_checkpoint=True)
            elif self.cfg.general.continue_training_mode == "best":
                self.trainer.train(
                    resume_from_checkpoint=os.path.join(
                        self.cfg.general.output_dir, "best_model"
                    )
                )
            elif self.cfg.general.continue_training_mode == "best_metric":
                raise NotImplementedError(
                    "Currently not supported to continue from best metric!"
                    " Implement yourselve or just set `best_metric` to "
                    "`best_model`."
                )

        else:
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
        if os.getenv("WANDB_MODE", "online").lower() == "offline" and wandb.run is not None:
            final_config = dict(wandb.config)
            logger.info_main(f"Final config verification - Total keys: {len(final_config)}")
            if len(final_config) < 5:  # Arbitrary threshold - expect at least some config
                logger.warning_main(
                    "Suspiciously few config keys logged - may indicate config persistence issue"
                )

        if is_main_process():
            wandb.finish()

        if self.is_distributed:
            dist.barrier()

        log_banner("Successfully finished Training")


def setup_config_and_logging() -> RunConfig:
    args_parser = argparse.ArgumentParser(description="Train a Vision Language Model")

    args_parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Path to config file (YAML or JSON), relative to the 'VLM_CONFIG_DIR' environment variable",
    )

    args_parser.add_argument(
        "--add_base_configs",
        type=str,
        nargs="+",
        default=[],
        help="Optional list of base config files",
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

    # Set to absolute path
    cfg_dict["general"]["train_config"] = str(train_config_path)

    # Parse all args into their respective Pydantic models/configs
    cfg = RunConfig.model_validate(cfg_dict)

    # Setup output directory and log file
    if is_main_process():
        os.makedirs(cfg.general.output_dir, exist_ok=True)
        shutil.copyfile(
            cfg.general.train_config, os.path.join(cfg.general.output_dir, "input_config.yml")
        )

    log_file_path = os.path.join(cfg.general.output_dir, "train.log")
    setup_logger(log_file_path)  # Setup logger on every rank

    logger.info_main(f"Successfully set up output directory at {cfg.general.output_dir}")
    print(f"Successfully set up output directory at {cfg.general.output_dir}")

    cfg.merge_subconfigs_with_cli_args()
    cfg.apply_conditional_resets()

    # Log the final configuration if we are not continuing training
    if is_main_process():
        if not cfg.general.continue_training:
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
                logger.info_main(
                    f"Offline mode - ensuring config persistence. Total config keys: {len(config_dict)}"
                )

                # Re-update the config to ensure it's properly saved in offline mode
                wandb.config.update(config_dict, allow_val_change=True)

                # Save config to a backup JSON file in the run directory for extra safety
                if hasattr(wandb.run, "dir") and wandb.run.dir:
                    config_backup_path = os.path.join(wandb.run.dir, "config_backup.json")
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
    vlm_model: AbstractVLMModel, cfg: RunConfig, world_size: int = 1, rank: int = 0
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


def train_val_data_from_config(cfg: RunConfig):
    if cfg.model.vision_encoder:
        vision_encoder_patch_size = cfg.model.vision_encoder.patch_size
    else:
        vision_encoder_patch_size = None

    do_validation = cfg.data.val_dataset_json_path is not None

    # Create datasets
    if cfg.data.dataset_class_name:
        dataset_class = cfg.data.get_dataset_class()
    elif cfg.data.dataset_name == "amosmm":
        # Use AMOSMM dataset
        dataset_class = AMOSMMDataset
    elif cfg.data.dataset_name == "ctrate":
        # Use CTRATE dataset
        if cfg.data.retrieval_json_path is not None:
            if cfg.data.retrieval_use_image_in_prompt:
                dataset_class = CTRATEDatasetRetrievalWithImage
            else:
                dataset_class = CTRATEDatasetRetrieval
        else:
            dataset_class = CTRATEDataset
    elif cfg.data.dataset_name == "merlin":
        if cfg.data.retrieval_json_path is not None:
            if cfg.data.retrieval_use_image_in_prompt:
                dataset_class = MerlinDatasetRetrievalWithImage
            else:
                dataset_class = MerlinDatasetRetrieval
        else:
            dataset_class = MerlinDataset
    elif cfg.data.dataset_name == "toothfairy4":
        dataset_class = ToothFairy4Dataset
    else:
        raise ValueError(f"Unknown dataset name: {cfg.data.dataset_name}")

    train_dataset = dataset_class(
        cfg=cfg,
        split="train",
        vision_encoder_patch_size=vision_encoder_patch_size,
    )

    val_dataset = dataset_class(
        cfg=cfg,
        split="val",
        vision_encoder_patch_size=vision_encoder_patch_size,
    ) if do_validation else None

    if cfg.data.embedding_constant is not None:
        logger.info_main(f"Using embedding constant: {cfg.data.embedding_constant}")
        print(f"Using embedding constant: {cfg.data.embedding_constant}")

    logger.info_main(
        f"Loaded {len(train_dataset)} training samples and "
        f"{len(val_dataset) if val_dataset is not None else 'no'} validation samples"
    )
    logger.info_main(f"Using train batch size: {cfg.trainer.per_device_train_batch_size}")
    if val_dataset is not None:
        logger.info_main(f"Using eval batch size: {cfg.trainer.per_device_eval_batch_size}")

    return train_dataset, val_dataset


@logger.catch(reraise=True)
def main():
    setup_ddp()  # setup DDP first so we can communicate shared output folder
    cfg = setup_config_and_logging()

    seed_everything(cfg.general.seed)  # Set a random seed for reproducibility

    try:
        vlm_model_class = VLM_REGISTRY[cfg.model.model_type].model_class
    except KeyError:
        raise ValueError(f"Unknown model type: {cfg.general.model_type}")

    if cfg.general.continue_training:
        cfg_continue_training_mode = cfg.general.continue_training_mode
        vlm_model, cfg, ckpt_dir = AbstractVLMModel.load_from_run_dir(
            cfg.general.output_dir,
            mode=cfg.general.continue_training_mode
        )
        cfg.general.continue_training_mode = cfg_continue_training_mode
        cfg.general.continue_training = True
    else:
        vlm_model = vlm_model_class(
            model_cfg=cfg.model,
            vision_input_type=cfg.data.vision_input_type,
            vision_embed_dim=cfg.get_vision_embed_dim(),
            vision_embed_size=cfg.get_vision_embed_size(),
            text_only=cfg.data.text_only,
            do_truncation=cfg.data.do_truncation,
            truncate_after=cfg.data.truncate_after,
            truncate_at=cfg.data.truncate_at,
            local_files_only=cfg.general.local_files_only
        )

    if cfg.model.vision_encoder:
        vision_encoder_patch_size = cfg.model.vision_encoder.patch_size

    train_dataset, val_dataset = train_val_data_from_config(cfg)

    # Create the trainer and train the model
    trainer = UnifiedVLMTrainer(vlm_model=vlm_model, cfg=cfg)

    trainer.train(train_dataset, val_dataset)
    logger.remove()

    if cfg.eval.run_inference:
        inference_args = argparse.Namespace(
            run_dir=cfg.general.output_dir,
            mode="final"
        )
        inference_main(inference_args, is_main=False)


if __name__ == "__main__":
    main()
