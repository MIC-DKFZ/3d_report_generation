import argparse
import os
import shutil
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig

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
from vlm.data.samplers import EvalDistributedSampler
from inference import main as inference_main
from vlm.models.abstract_vl_wrapper import AbstractVLWrapper
from vlm.training.callbacks.logging import LossLoggingCallback, PycocoevalCallback, WandbGroupedLRCallback, \
    TerminateOnNaNLossCallback, EveryNEpochsCallback
from vlm.utils.training.checkpointing import (
    CheckpointPlan,
    TrainRuntimeArgs,
    create_checkpoint_plan,
    load_run_config_from_run_dir,
    restore_model_checkpoints,
    save_final_config,
)
from vlm.training.trainers import VLMSFTTrainer
from vlm.registries.vlm_registry import VLM_REGISTRY
from vlm.utils import logger
from vlm.utils.ddp_utils import setup_ddp, is_main_process
from vlm.utils.seeding import seed_everything
from vlm.utils.data.dataset_utils import collect_image_ids_from_config_dataset_jsons
from vlm.utils.training.optimizer_and_params import count_parameters
from vlm.utils.training.wandb_utils import finish_wandb, initialize_wandb
from vlm.utils.data.local_staging import stage_vision_input_to_local
from vlm.utils.load_and_save_files import load_config_file, save_json
from vlm.utils.logging import setup_logger, log_banner
from vlm.utils.memory_utils import (
    aggressive_cleanup,
    log_memory_usage,
    setup_memory_efficient_training,
)
from vlm.utils.paths import VLM_CONFIG_DIR, resolve_run_dir


INPUT_CONFIG_FILENAME = "input_config.yml"
TRAIN_LOG_FILENAME = "train.log"
ADAPTER_STATS_FILENAME = "adapter_stats.json"


class UnifiedVLMTrainer:
    def __init__(
        self,
        vlm_model: AbstractVLWrapper,
        cfg: RunConfig,
        checkpoint_plan: CheckpointPlan,
        ddp_setup: tuple[bool, int, int, int, int],
    ):
        """
        Initialize the unified VLM trainer.

        Args:
            vlm_model: An instance of a class derived from AbstractVLWrapper
            cfg: The RunConfig of the experiment
            checkpoint_plan: Ordered model and training-state checkpoint sources
            ddp_setup: Distributed topology initialized by main
        """
        self.vlm_model: AbstractVLWrapper = vlm_model
        self.trainer: VLMSFTTrainer | None = None
        self.cfg: RunConfig = cfg
        self.checkpoint_plan = checkpoint_plan

        (
            self.is_distributed,
            self.rank,
            self.world_size,
            self.local_rank,
            self.local_world_size,
        ) = ddp_setup

    def get_training_args(self):
        """Configure standard training arguments for distributed training"""
        model_args = self.vlm_model.get_model_specific_training_args()

        trainer_args = self.cfg.trainer.model_dump(
            exclude={"early_stopping_patience", "early_stopping_threshold"}
        )

        return SFTConfig(
            **trainer_args,
            **model_args,
            run_name=self.cfg.general.run_name,
            output_dir=self.cfg.general.output_dir,
            seed=self.cfg.general.seed,
        )

    def train(self, train_dataset, val_dataset, callback_val_dataset=None) -> bool:
        """
        Main training function that trains the VLM model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset used by the trainer for eval loss.
            callback_val_dataset: Validation dataset used by custom generation callbacks.

        Returns:
            Whether the resumed run had already reached its configured training limit.
        """
        if self.is_distributed:
            dist.barrier()

        # Setup memory efficient training
        setup_memory_efficient_training()
        if is_main_process():
            log_memory_usage("Before Training Setup")

        # Assemble the model and prepare PEFT/trainability before restoring weights.
        model = self.vlm_model.initialize_model()

        # Sparse checkpoints are applied in dependency order; resume weights come last.
        restore_model_checkpoints(self.vlm_model, self.checkpoint_plan)

        # Log adapter-specific numbers (e.g. number of params)
        if self.vlm_model.vision_adapter is not None:
            adapter_parameter_count = count_parameters(self.vlm_model.vision_adapter)
            _stats = {
                "parameter_count": adapter_parameter_count
            }
            save_json(_stats, Path(self.cfg.general.output_dir, ADAPTER_STATS_FILENAME))

        initialize_wandb(self.vlm_model, self.cfg, self.world_size, self.rank)

        # Configure training arguments
        training_args = self.get_training_args()

        callbacks: list[TrainerCallback] = [
            LossLoggingCallback(),
            WandbGroupedLRCallback(),
            TerminateOnNaNLossCallback(threshold=3)
        ]

        if self.cfg.trainer.eval_strategy == "epoch":
            if self.cfg.trainer.eval_steps is None:
                raise ValueError(
                    "trainer.eval_steps must be set when trainer.eval_strategy='epoch' "
                    "because it controls EveryNEpochsCallback.n."
                )
            callbacks.append(EveryNEpochsCallback(n=self.cfg.trainer.eval_steps))

        if val_dataset is not None:
            callback_eval_dataset = callback_val_dataset if callback_val_dataset is not None else val_dataset
            if self.is_distributed:
                sampler = EvalDistributedSampler(
                    callback_eval_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False
                )
            else:
                sampler = None

            callback_eval_dataloader = DataLoader(
                callback_eval_dataset,
                batch_size=self.cfg.eval.per_device_batch_size,
                shuffle=False,
                sampler=sampler,
                collate_fn=self.vlm_model.collate_fn,
                num_workers=self.cfg.trainer.dataloader_num_workers,
                # IMPORTANT: Keep pin_memory disabled for the callback dataloader. Tests with 1000
                # callback samples showed that enabling pinned memory can lead to
                # multi-minute stalls in the main training loop after the callback finishes.
                pin_memory=False
            )

            callbacks.append(
                PycocoevalCallback(  # type: ignore[arg-type]
                    eval_dataloader=callback_eval_dataloader,
                    max_new_tokens=self.cfg.eval.max_new_tokens,
                    eval_every_n_epochs=self.cfg.eval.eval_every_n_epochs,
                    tokenizer=self.vlm_model.tokenizer,
                    processor=self.vlm_model.processor,
                    is_distributed=self.is_distributed,
                    calculate_radbert_score=self.cfg.eval.calculate_radbert_score,
                    dataset_name=self.cfg.data.dataset_name
                )
            )

        if self.cfg.general.do_early_stopping:
            callbacks.append(
                EarlyStoppingCallback(  # type: ignore
                    early_stopping_patience=self.cfg.trainer.early_stopping_patience,
                    early_stopping_threshold=self.cfg.trainer.early_stopping_threshold,
                )
            )

        # Initialize SFT trainer
        self.trainer = VLMSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.vlm_model.collate_fn,
            callbacks=callbacks,
            optimizer_scheduler_checkpoint=self.checkpoint_plan.optimizer_scheduler_checkpoint,
            model_weights_preloaded=self.checkpoint_plan.is_resume,
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

        # Model weights are already loaded; pass the checkpoint path so Trainer can restore
        # its step, callback, scaler, RNG, and dataloader progress when continuing a run.
        resume_checkpoint = self.checkpoint_plan.trainer_state_checkpoint
        self.trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
        )
        already_complete_on_resume = self.trainer.already_complete_on_resume

        if is_main_process():
            log_memory_usage("After Training")

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

        finish_wandb()

        if self.is_distributed:
            dist.barrier()

        log_banner("Successfully finished Training")
        return already_complete_on_resume


def resolve_dataloader_workers(
    cfg: RunConfig, is_distributed: bool, local_world_size: int
) -> None:
    num_cores = os.cpu_count() or 1
    cfg.trainer.dataloader_num_workers = (
        max(1, (num_cores // 4) // local_world_size) if is_distributed else 2
    )


def setup_config_and_logging(
    is_distributed: bool, local_world_size: int
) -> tuple[RunConfig, TrainRuntimeArgs]:
    args_parser = argparse.ArgumentParser(
        description="Train a Vision Language Model", allow_abbrev=False
    )

    args_parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help="Path to config file (YAML or JSON), relative to the 'VLM_CONFIG_DIR' environment variable",
    )
    args_parser.add_argument(
        "--add_base_configs",
        type=str,
        nargs="+",
        default=[],
        help="Optional list of base config files",
    )

    args_parser.add_argument(
        "--continue_training_run_dir",
        type=Path,
        default=None,
        help="Run directory to continue training from.",
    )

    args, config_cli_args = args_parser.parse_known_args()

    if args.continue_training_run_dir is None and args.train_config is None:
        args_parser.error("--train_config or --continue_training_run_dir is required.")

    runtime_args = TrainRuntimeArgs(
        continue_training_run_dir=(
            resolve_run_dir(args.continue_training_run_dir)
            if args.continue_training_run_dir is not None
            else None
        ),
    )

    if runtime_args.continue_training_run_dir is not None:
        cfg, config_warnings = load_run_config_from_run_dir(runtime_args.continue_training_run_dir)

        log_file_path = os.path.join(cfg.general.output_dir, TRAIN_LOG_FILENAME)
        setup_logger(log_file_path)  # Setup logger on every rank

        for message in config_warnings:
            logger.warning_main(message)

        logger.info_main(f"Continuing training from run directory: {cfg.general.output_dir}")

        cfg.merge_subconfigs_with_cli_args(config_cli_args)
        cfg.general.output_dir = str(runtime_args.continue_training_run_dir)
        cfg.apply_conditional_resets()
        resolve_dataloader_workers(cfg, is_distributed, local_world_size)

        # Persist CLI overrides for future resumptions.
        if is_main_process():
            save_final_config(cfg)

        return cfg, runtime_args

    assert args.train_config is not None
    train_config_path = Path(VLM_CONFIG_DIR, args.train_config)
    if not train_config_path.exists():
        raise ValueError(f"The provided config file does not exist: {train_config_path}")

    cfg_dict = load_config_file(train_config_path)

    if args.add_base_configs:
        # Give base configs that are passed via CLI higher prio
        cfg_dict["base_configs"] = args.add_base_configs + (cfg_dict.get("base_configs") or [])

    # Merge base configs into cfg_dict
    cfg_dict = RunConfig.apply_base_configs(cfg_dict=cfg_dict, config_dir=VLM_CONFIG_DIR)

    # Set to absolute path
    cfg_dict["general"]["train_config"] = str(train_config_path)

    # Validate the file config before registering CLI override options.
    cfg = RunConfig.model_validate(cfg_dict)

    # Setup output directory and log file
    if is_main_process():
        os.makedirs(cfg.general.output_dir, exist_ok=True)
        shutil.copyfile(
            cfg.general.train_config, os.path.join(cfg.general.output_dir, INPUT_CONFIG_FILENAME)
        )

    log_file_path = os.path.join(cfg.general.output_dir, TRAIN_LOG_FILENAME)
    setup_logger(log_file_path)  # Setup logger on every rank

    logger.info_main(f"Successfully set up output directory at {cfg.general.output_dir}")
    print(f"Successfully set up output directory at {cfg.general.output_dir}")

    cfg.merge_subconfigs_with_cli_args(config_cli_args)
    cfg.apply_conditional_resets()
    resolve_dataloader_workers(cfg, is_distributed, local_world_size)

    if is_main_process():
        save_final_config(cfg)

    return cfg, runtime_args


def train_val_data_from_config(cfg: RunConfig):
    vision_encoder_patch_size = cfg.model.vision_encoder.patch_size if cfg.model.vision_encoder else None

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

    train_dataset = dataset_class(  # type: ignore
        cfg=cfg,
        split="train",
        n_samples=cfg.data.n_train_samples,
        vision_encoder_patch_size=vision_encoder_patch_size,
    )

    val_dataset = dataset_class(  # type: ignore
        cfg=cfg,
        split="val",
        n_samples=cfg.data.n_val_samples,
        vision_encoder_patch_size=vision_encoder_patch_size,
    ) if do_validation else None

    callback_val_dataset = None
    if val_dataset:
        callback_val_dataset = dataset_class(  # type: ignore
            cfg=cfg,
            split="val",
            n_samples=cfg.eval.callback_n_samples,
            vision_encoder_patch_size=vision_encoder_patch_size,
        )

    if cfg.data.embedding_constant is not None:
        logger.info_main(f"Using embedding constant: {cfg.data.embedding_constant}")
        print(f"Using embedding constant: {cfg.data.embedding_constant}")

    logger.info_main(
        f"Loaded {len(train_dataset)} training samples and "
        f"{len(val_dataset) if val_dataset is not None else 'no'} validation samples"
    )
    if callback_val_dataset is not None:
        logger.info_main(
            f"Loaded {len(callback_val_dataset)} validation samples for callback evaluation"
        )
    logger.info_main(f"Using train batch size: {cfg.trainer.per_device_train_batch_size}")
    if val_dataset is not None:
        logger.info_main(f"Using eval batch size: {cfg.trainer.per_device_eval_batch_size}")

    return train_dataset, val_dataset, callback_val_dataset


@logger.catch(reraise=True)
def main():
    # Set up DDP before config validation so generated run paths can be shared across ranks.
    ddp_setup = setup_ddp()
    cfg, runtime_args = setup_config_and_logging(
        is_distributed=ddp_setup[0], local_world_size=ddp_setup[4]
    )
    checkpoint_plan = create_checkpoint_plan(cfg, runtime_args)

    # Pass the actual run directory to the submission script for subsequent (test-set) inference.
    run_dir_file = os.getenv("VLM_RUN_DIR_FILE")
    if run_dir_file and is_main_process():
        run_dir_file_path = Path(run_dir_file)
        run_dir_file_path.parent.mkdir(parents=True, exist_ok=True)
        run_dir_file_path.write_text(cfg.general.output_dir + "\n", encoding="utf-8")

    # Copy vision inputs to node-local storage to reduce shared filesystem I/O.
    if cfg.general.stage_vision_input_to_local:
        image_ids = collect_image_ids_from_config_dataset_jsons(cfg, ["train", "val", "test"])
        cfg.data.vision_input_path = str(
            stage_vision_input_to_local(
                vision_input_path=cfg.data.vision_input_path,  # type: ignore
                vision_input_is_sharded=cfg.data.vision_input_is_sharded,
                image_ids=image_ids,
            )
        )

    seed_everything(cfg.general.seed)  # Set a random seed for reproducibility

    try:
        vlm_model_class = VLM_REGISTRY[cfg.model.model_type].model_class
    except KeyError:
        raise ValueError(f"Unknown model type: {cfg.model.model_type}")

    vlm_model = vlm_model_class(
        model_cfg=cfg.model,
        vision_input_type=cfg.data.vision_input_type,
        vision_embed_dim=cfg.get_vision_embed_dim(),
        vision_embed_size=cfg.get_vision_embed_size(),
        text_only=cfg.data.text_only,
        max_response_tokens=cfg.data.max_response_tokens,
        local_files_only=cfg.general.local_files_only
    )

    train_dataset, val_dataset, callback_val_dataset = train_val_data_from_config(cfg)

    # Create the trainer and train the model
    trainer = UnifiedVLMTrainer(
        vlm_model=vlm_model,
        cfg=cfg,
        checkpoint_plan=checkpoint_plan,
        ddp_setup=ddp_setup,
    )

    already_complete_on_resume = trainer.train(train_dataset, val_dataset, callback_val_dataset)
    logger.remove()

    if cfg.eval.run_inference and not already_complete_on_resume:
        inference_args = argparse.Namespace(
            run_dir=cfg.general.output_dir,
            mode=cfg.eval.mode
        )
        inference_main(inference_args, vision_input_path=cfg.data.vision_input_path, is_main=False)


if __name__ == "__main__":
    main()
