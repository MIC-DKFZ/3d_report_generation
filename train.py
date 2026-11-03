import argparse
import os
import shutil
from pathlib import Path

import torch.distributed as dist
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

from vlm.configs.run_config import RunConfig
from vlm.data.dataloaders import create_eval_dataloader
from vlm.data.factory import create_dataset
from inference import main as inference_main
from vlm.models.abstract_vl_wrapper import AbstractVLWrapper
from vlm.models.factory import create_model_wrapper
from vlm.training.callbacks import (
    EveryNEpochsCallback,
    GracefulStopCallback,
    LossLoggingCallback,
    ReportEvaluationCallback,
    TerminateOnNaNLossCallback,
    WandbGroupedLRCallback,
)
from vlm.training.checkpointing import (
    CheckpointPlan,
    TrainRuntimeArgs,
    create_checkpoint_plan,
    load_run_config_from_run_dir,
    restore_model_checkpoints,
    save_final_config,
)
from vlm.training.phases import train_phases
from vlm.training.trainers import VLMTrainer
from vlm.utils import logger
from vlm.utils.ddp import setup_ddp, is_main_process
from vlm.utils.seeding import seed_everything
from vlm.data.image_ids import collect_image_ids_from_config_dataset_jsons
from vlm.training.parameters import count_parameters
from vlm.training.wandb import finish_wandb, initialize_wandb
from vlm.data.staging import stage_vision_input_to_local
from vlm.data.vision_input_settings import apply_saved_settings
from vlm.utils.serialization import load_config_file, save_json
from vlm.utils.logging import setup_logger, log_banner
from vlm.utils.memory import (
    aggressive_cleanup,
    log_memory_usage,
    setup_memory_efficient_training,
)
from vlm.utils.paths import resolve_run_dir, get_vlm_config_dir


INPUT_CONFIG_FILENAME = "input_config.yml"
TRAIN_LOG_FILENAME = "train.log"
ADAPTER_STATS_FILENAME = "adapter_stats.json"
# Run directory markers for follow-up jobs: training finished, then inference finished too.
TRAINING_COMPLETE_FILENAME = "TRAINING_COMPLETE"
RUN_COMPLETE_FILENAME = "RUN_COMPLETE"


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
        self.cfg: RunConfig = cfg
        self.checkpoint_plan = checkpoint_plan

        (
            self.is_distributed,
            self.rank,
            self.world_size,
            self.local_rank,
            self.local_world_size,
        ) = ddp_setup
        # Whether the last 'train' call saved and stopped before the end, for a follow-up job.
        self.stopped_early = False

    def get_training_args(self):
        """Configure standard training arguments for distributed training"""
        model_args = self.vlm_model.get_model_specific_training_args()

        trainer_args = self.cfg.trainer.model_dump(
            exclude={
                "early_stopping_patience",
                "early_stopping_threshold",
                "deepspeed",
                "gradient_checkpointing_fraction",
            }
        )
        deepspeed = self.cfg.trainer.deepspeed

        return TrainingArguments(
            **trainer_args,
            **model_args,
            deepspeed=deepspeed.get_resolved_path() if deepspeed is not None else None,
            run_name=self.cfg.general.run_name,
            output_dir=self.cfg.general.output_dir.get_resolved_path(),
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
            save_json(
                _stats, Path(self.cfg.general.output_dir.get_resolved_path(), ADAPTER_STATS_FILENAME)
            )

        initialize_wandb(self.vlm_model, self.cfg, self.world_size, self.rank)

        callbacks: list[TrainerCallback] = [
            LossLoggingCallback(),
            WandbGroupedLRCallback(),
            TerminateOnNaNLossCallback(threshold=3)
        ]

        if self.cfg.trainer.eval_strategy == "epoch":
            callbacks.append(EveryNEpochsCallback(n=self.cfg.eval.eval_every_n_epochs))

        if val_dataset is not None:
            callback_eval_dataset = callback_val_dataset if callback_val_dataset is not None else val_dataset
            callback_eval_dataloader = create_eval_dataloader(
                callback_eval_dataset,
                batch_size=self.cfg.eval.per_device_batch_size,
                collate_fn=self.vlm_model.collate_fn,
                num_workers=self.cfg.trainer.dataloader_num_workers,
                prefetch_factor=self.cfg.trainer.dataloader_prefetch_factor,
                persistent_workers=self.cfg.trainer.dataloader_persistent_workers,
                is_distributed=self.is_distributed,
                world_size=self.world_size,
                rank=self.rank,
            )

            callbacks.append(
                ReportEvaluationCallback(  # type: ignore[arg-type]
                    eval_dataloader=callback_eval_dataloader,
                    max_new_tokens=self.cfg.eval.max_new_tokens,
                    generation_kwargs=self.cfg.eval.generation_kwargs(),
                    tokenizer=self.vlm_model.tokenizer,
                    is_distributed=self.is_distributed,
                    calculate_classifier_score=self.cfg.eval.calculate_classifier_score,
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

        # Last, after 'EveryNEpochsCallback' has decided which epochs evaluate.
        stop_callback = GracefulStopCallback.from_environment(
            self.cfg.general.output_dir.get_resolved_path()
        )
        callbacks.append(stop_callback)

        # Created after the callbacks: with DeepSpeed ZeRO-3, models loaded once the
        # TrainingArguments exist are sharded across ranks.
        training_args = self.get_training_args()

        scheduler = self.cfg.scheduler.to_instance(
            module_attr_map=self.vlm_model.module_attr_map,
            forbidden_name_patterns_for_wd=self.vlm_model.forbidden_name_patterns_for_wd,
            output_dir=self.cfg.general.output_dir.get_resolved_path()
        )
        phases = scheduler.phases(self.cfg.trainer.num_train_epochs)
        if len(phases) > 1 and training_args.max_steps > 0:
            raise ValueError(
                "Training phases are whole epochs; set 'trainer.num_train_epochs' instead of "
                "'trainer.max_steps'."
            )
        if len(phases) > 1 and training_args.load_best_model_at_end:
            raise ValueError(
                "'trainer.load_best_model_at_end' would load the best checkpoint at the end of "
                "every training phase instead of continuing from the phase's final weights."
            )

        def create_trainer(phase, phase_start_state):
            scheduler.set_trainable_params(model, phase)
            return VLMTrainer(
                model=model,
                args=training_args,
                processing_class=self.vlm_model.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=self.vlm_model.collate_fn,
                callbacks=callbacks,
                scheduler=scheduler,
                phase=phase,
                phase_start_state=phase_start_state,
                model_weights_preloaded=self.checkpoint_plan.is_resume,
                gradient_checkpointing_fraction=self.cfg.trainer.gradient_checkpointing_fraction,
            )

        if self.is_distributed:
            dist.barrier()

        # Train model
        if is_main_process():
            log_memory_usage("Before Training")

        # Model weights are already loaded; the checkpoint restores the trainer's step, callback,
        # scaler, RNG, dataloader, and (inside a phase) optimizer progress when continuing a run.
        already_complete_on_resume = not train_phases(
            phases,
            create_trainer,
            self.checkpoint_plan.trainer_state_checkpoint,
            stop_requested=lambda: stop_callback.stopped,
        )
        self.stopped_early = stop_callback.stopped_before_end
        # Inference loads the model again and must not shard it as under DeepSpeed ZeRO-3.
        unset_hf_deepspeed_config()

        if is_main_process():
            log_memory_usage("After Training")

        # Synchronize all processes before cleanup
        if self.is_distributed:
            dist.barrier()

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

    if runtime_args.continue_training_run_dir is None:
        assert args.train_config is not None

        config_dir = get_vlm_config_dir()
        train_config_path = Path(config_dir, args.train_config)
        if not train_config_path.exists():
            raise ValueError(f"The provided config file does not exist: {train_config_path}")

        cfg_dict = load_config_file(train_config_path)

        if args.add_base_configs:
            # Give base configs that are passed via CLI higher prio
            cfg_dict["base_configs"] = args.add_base_configs + (cfg_dict.get("base_configs") or [])

        # Merge base configs into cfg_dict
        cfg_dict = RunConfig.apply_base_configs(cfg_dict=cfg_dict, config_dir=config_dir)

        cfg_dict.setdefault("general", {})["train_config"] = str(train_config_path)

        # Apply CLI overrides before creating the output directory, which they can change.
        cfg = RunConfig.validate_with_cli_args(cfg_dict, config_cli_args)

        output_dir = cfg.general.output_dir.get_resolved_path()
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
            shutil.copyfile(cfg.general.train_config, Path(output_dir, INPUT_CONFIG_FILENAME))
        setup_logger(os.path.join(output_dir, TRAIN_LOG_FILENAME))
        logger.info_main(f"Successfully set up output directory at {output_dir}")

    else:
        cfg, config_warnings = load_run_config_from_run_dir(runtime_args.continue_training_run_dir)

        output_dir = cfg.general.output_dir.get_resolved_path()
        setup_logger(os.path.join(output_dir, TRAIN_LOG_FILENAME))

        for message in config_warnings:
            logger.warning_main(message)

        logger.info_main(f"Continuing training from run directory: {output_dir}")

        cfg.merge_subconfigs_with_cli_args(config_cli_args)
        cfg.general.output_dir.set_runtime_path(runtime_args.continue_training_run_dir)

    cfg.apply_conditional_resets()
    cfg.resolve_dataloader_workers(is_distributed, local_world_size)
    # Record how the vision inputs were made, before staging them changes their path.
    apply_saved_settings(cfg.data)

    if is_main_process():
        save_final_config(cfg)

    return cfg, runtime_args


def train_val_data_from_config(cfg: RunConfig):
    train_dataset = create_dataset(cfg, split="train", n_samples=cfg.data.n_train_samples)
    val_dataset = (
        create_dataset(cfg, split="val", n_samples=cfg.data.n_val_samples)
        if cfg.data.val_dataset_json_path is not None
        else None
    )
    callback_val_dataset = (
        create_dataset(cfg, split="val", n_samples=cfg.eval.callback_n_samples)
        if val_dataset
        else None
    )

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


def inference_is_pending(run_dir: Path, run_inference: bool, already_complete_on_resume: bool) -> bool:
    """
    Whether to run inference after training: always after training steps, and on a resume of a
    finished run only when an earlier job finished training but not its inference.
    """
    if not run_inference:
        return False
    if not already_complete_on_resume:
        return True
    return (run_dir / TRAINING_COMPLETE_FILENAME).exists() and not (
        run_dir / RUN_COMPLETE_FILENAME
    ).exists()


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
        run_dir_file_path.write_text(
            cfg.general.output_dir.get_resolved_path() + "\n", encoding="utf-8"
        )

    # Copy vision inputs to node-local storage to reduce shared filesystem I/O.
    if cfg.general.stage_vision_input_to_local:
        image_ids = collect_image_ids_from_config_dataset_jsons(cfg, ["train", "val", "test"])
        cfg.data.vision_input_path.set_runtime_path(
            stage_vision_input_to_local(
                vision_input_path=cfg.data.vision_input_path.get_resolved_path(),
                vision_input_is_sharded=cfg.data.vision_input_is_sharded,
                image_ids=image_ids,
            ),
        )

    seed_everything(cfg.general.seed)  # Set a random seed for reproducibility

    vlm_model = create_model_wrapper(cfg)

    train_dataset, val_dataset, callback_val_dataset = train_val_data_from_config(cfg)

    # Create the trainer and train the model
    trainer = UnifiedVLMTrainer(
        vlm_model=vlm_model,
        cfg=cfg,
        checkpoint_plan=checkpoint_plan,
        ddp_setup=ddp_setup,
    )

    already_complete_on_resume = trainer.train(train_dataset, val_dataset, callback_val_dataset)
    run_dir = Path(cfg.general.output_dir.get_resolved_path())
    if trainer.stopped_early:
        logger.info_main(f"Stopped before the end of training; continue with a run of {run_dir}.")
        logger.remove()
        return
    logger.remove()

    run_inference = inference_is_pending(run_dir, cfg.eval.run_inference, already_complete_on_resume)
    if is_main_process():
        (run_dir / TRAINING_COMPLETE_FILENAME).touch()

    if run_inference:
        inference_args = argparse.Namespace(
            run_dir=str(run_dir),
            mode=cfg.eval.mode
        )
        inference_main(
            inference_args,
            vision_input_path=(
                cfg.data.vision_input_path.get_resolved_path()
                if cfg.data.vision_input_path is not None else None
            ),
            is_main=False
        )
    if is_main_process():
        (run_dir / RUN_COMPLETE_FILENAME).touch()


if __name__ == "__main__":
    main()
