"""Run-level checkpoint resolution and training-state planning."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch.nn as nn
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME

from vlm.configs.run_config import RunConfig
from vlm.training.checkpoint_state import read_manifest
from vlm.utils import logger
from vlm.utils.paths import find_run_dir_in_output_dir
from vlm.utils.load_and_save_files import load_json, load_yaml, save_yaml

if TYPE_CHECKING:
    from vlm.models.abstract_vlm import AbstractVLMModel


FINAL_CONFIG_FILENAME = "final_config.yml"


@dataclass(frozen=True)
class TrainRuntimeArgs:
    continue_training_run_dir: Path | None


@dataclass(frozen=True)
class CheckpointPlan:
    """Ordered model-weight sources and optional training-state sources."""

    model_checkpoints: tuple[Path, ...] = ()
    trainer_state_checkpoint: Path | None = None
    optimizer_scheduler_checkpoint: Path | None = None

    @property
    def is_resume(self) -> bool:
        return self.trainer_state_checkpoint is not None


def resolve_ckpt_dir_in_run_dir(
    run_dir: str | Path,
    mode: Literal["final", "best", "best_metric"],
) -> Path:
    """Resolve a final or best checkpoint within a training run directory."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"The specified run directory does not exist: {run_dir}")
    if mode == "best_metric":
        best_metric_dirs = list(run_dir.glob("best_model-*"))
        if len(best_metric_dirs) != 1 or not best_metric_dirs[0].is_dir():
            raise FileNotFoundError(f"Expected exactly one 'best_model-*' directory in {run_dir}")
        return best_metric_dirs[0]

    checkpoint_dirs = [path for path in run_dir.glob("checkpoint-*") if path.is_dir()]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No HF trainer checkpoint directories found in {run_dir}")
    final_checkpoint = max(
        checkpoint_dirs,
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    if mode == "final":
        return final_checkpoint
    if mode == "best":
        trainer_state = load_json(final_checkpoint / TRAINER_STATE_NAME)
        best_checkpoint = Path(trainer_state["best_model_checkpoint"])
        if not best_checkpoint.is_dir():
            raise FileNotFoundError(
                f"The HF trainer 'best_model_checkpoint' directory does not exist in {run_dir}"
            )
        return best_checkpoint
    raise ValueError(f"Unknown checkpoint type to load: mode={mode}")


def load_run_config_from_run_dir(run_dir: str | Path) -> RunConfig:
    run_dir = Path(run_dir)
    final_config_path = run_dir / FINAL_CONFIG_FILENAME
    cfg_dict = load_yaml(final_config_path)["run"]
    cfg = RunConfig.model_validate(cfg_dict)

    # The CLI resume directory remains authoritative if the run was moved.
    cfg.general.output_dir = str(run_dir)
    return cfg


def save_final_config(cfg: RunConfig) -> None:
    """Persist the effective configuration used by this training invocation."""
    final_config_path = Path(cfg.general.output_dir, FINAL_CONFIG_FILENAME)
    save_yaml(
        {
            "command": " ".join(sys.argv),
            "run": cfg.model_dump(),
        },
        final_config_path,
        sort_keys=False,
    )


def _validated_optimizer_scheduler_checkpoint(
    checkpoint_dir: Path | None,
) -> Path | None:
    if checkpoint_dir is None:
        return None

    scheduler_path = checkpoint_dir / SCHEDULER_NAME
    optimizer_path = checkpoint_dir / OPTIMIZER_NAME
    if scheduler_path.is_file() and optimizer_path.is_file():
        return checkpoint_dir

    logger.warning_main(
        f"Skipping optimizer/scheduler initialization from checkpoint '{checkpoint_dir}' because "
        f"'{SCHEDULER_NAME}' or '{OPTIMIZER_NAME}' is missing."
    )
    return None


def create_checkpoint_plan(
    cfg: RunConfig,
    runtime_args: TrainRuntimeArgs,
) -> CheckpointPlan:
    """Resolve a checkpoint path or run name, plus resume state, preserving model-delta load order.

    ``init_from_checkpoint`` may be an absolute path to a checkpoint directory or a run
    directory name under ``VLM_OUTPUT_DIR``. Run names resolve to that run's final checkpoint.
    """
    init_checkpoint = None
    if cfg.general.init_from_checkpoint:
        init_source = Path(cfg.general.init_from_checkpoint)
        if init_source.is_absolute():
            init_checkpoint = init_source
            if not init_checkpoint.is_dir():
                raise FileNotFoundError(
                    f"The specified checkpoint directory does not exist: {init_checkpoint}"
                )
        else:
            run_dir = find_run_dir_in_output_dir(str(init_source))
            init_checkpoint = resolve_ckpt_dir_in_run_dir(run_dir, mode="final")
    model_checkpoints = (init_checkpoint,) if init_checkpoint is not None else ()

    if runtime_args.continue_training_run_dir is not None:
        resume_checkpoint = resolve_ckpt_dir_in_run_dir(
            run_dir=runtime_args.continue_training_run_dir,
            mode="final",
        )
        cumulative = read_manifest(resume_checkpoint) is not None
        return CheckpointPlan(
            # Cumulative checkpoints never depend on the initialization run.
            # Retain the old layering behavior only for legacy checkpoints.
            model_checkpoints=(resume_checkpoint,) if cumulative
            else model_checkpoints + (resume_checkpoint,),
            trainer_state_checkpoint=resume_checkpoint,
            optimizer_scheduler_checkpoint=_validated_optimizer_scheduler_checkpoint(
                resume_checkpoint
            ),
        )

    optimizer_scheduler_checkpoint = None
    if cfg.general.init_optimizer_scheduler_from_checkpoint:
        optimizer_scheduler_checkpoint = _validated_optimizer_scheduler_checkpoint(
            init_checkpoint
        )

    return CheckpointPlan(
        model_checkpoints=model_checkpoints,
        optimizer_scheduler_checkpoint=optimizer_scheduler_checkpoint,
    )


def log_checkpoint_load_result(
    model: nn.Module | None,
    checkpoint_dir: str | Path,
    missing_keys: list[str],
    unexpected_keys: list[str],
    preview_limit: int = 20,
) -> None:
    """Report checkpoint key mismatches, distinguishing trainable from frozen parameters."""
    if model is None:
        logger.warning_main("Checkpoint load: could not classify missing keys because model is None")
        trainable_names = set()
    else:
        trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}

    missing_trainable = [key for key in missing_keys if key in trainable_names]
    missing_frozen = [key for key in missing_keys if key not in trainable_names]
    for keys, description, log in (
        (missing_trainable, "trainable keys are missing from", logger.warning_main),
        (missing_frozen, "non-trainable keys are missing from", logger.info_main),
        (unexpected_keys, "unexpected keys in", logger.warning_main),
    ):
        if not keys:
            continue
        preview = keys[:preview_limit]
        suffix = " ..." if len(keys) > preview_limit else ""
        note = (
            " (usually expected for trainable-only checkpoints)"
            if keys is missing_frozen
            else ""
        )
        log(
            f"Checkpoint load: {len(keys)} {description} {checkpoint_dir}{note}. "
            f"Preview (first {len(preview)}): {preview}{suffix}"
        )

    if not missing_keys and not unexpected_keys:
        logger.info_main(
            f"Checkpoint load: no missing/unexpected keys when initializing from {checkpoint_dir}"
        )


def restore_model_checkpoints(
    vlm_model: "AbstractVLMModel",
    checkpoint_plan: CheckpointPlan,
) -> None:
    """Apply sparse model checkpoints in dependency order."""
    for checkpoint_dir in checkpoint_plan.model_checkpoints:
        logger.info_main(f"Loading model weights from {checkpoint_dir}")
        missing_keys, unexpected_keys = vlm_model.load_weights_from_checkpoint_dir(checkpoint_dir)
        log_checkpoint_load_result(
            model=vlm_model.model,
            checkpoint_dir=checkpoint_dir,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
