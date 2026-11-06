"""Run-level checkpoint resolution and training-state planning."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch.nn as nn
from transformers.trainer import TRAINER_STATE_NAME

from vlm.configs.run_config import RunConfig
from vlm.configs.compatibility import normalize_saved_config
from vlm.training.checkpoint_state import read_manifest
from vlm.utils import logger
from vlm.utils.paths import find_run_dir_in_output_dir, resolve_run_dir
from vlm.utils.serialization import load_json, load_yaml, save_yaml

if TYPE_CHECKING:
    from vlm.models.abstract_vl_wrapper import AbstractVLWrapper


FINAL_CONFIG_FILENAME = "final_config.yml"


@dataclass(frozen=True)
class TrainRuntimeArgs:
    continue_training_run_dir: Path | None


@dataclass(frozen=True)
class CheckpointPlan:
    """Ordered model-weight sources and optional training-state sources."""

    model_checkpoints: tuple[Path, ...] = ()
    trainer_state_checkpoint: Path | None = None

    @property
    def is_resume(self) -> bool:
        return self.trainer_state_checkpoint is not None


def resolve_ckpt_dir_in_run_dir(
    run_dir: str | Path,
    mode: Literal["final", "best"],
) -> Path:
    """Resolve a final or best checkpoint within a training run directory."""
    run_dir = resolve_run_dir(run_dir)
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


def load_run_config_from_run_dir(run_dir: str | Path) -> tuple[RunConfig, list[str]]:
    """Load a saved run config and return compatibility warnings for deferred logging."""
    run_dir = resolve_run_dir(run_dir)
    final_config_path = run_dir / FINAL_CONFIG_FILENAME
    cfg_dict = load_yaml(final_config_path)["run"]
    cfg_dict, messages = normalize_saved_config(cfg_dict)
    cfg = RunConfig.model_validate(cfg_dict)

    # The CLI resume directory remains authoritative if the run was moved.
    cfg.general.output_dir.set_runtime_path(run_dir)
    return cfg, messages


def _get_git_state() -> dict | None:
    """Return the commit, branch and dirty flag of the code repository, or None if unavailable."""
    repo_dir = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except FileNotFoundError:
        # No git executable, e.g. on cluster compute nodes: read the commit from the repository.
        state = _read_git_head(repo_dir)
        if state is None:
            logger.warning_main(
                f"Could not determine git state for '{FINAL_CONFIG_FILENAME}': no git executable "
                "and no readable git repository."
            )
        return state
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning_main(f"Could not determine git state for '{FINAL_CONFIG_FILENAME}': {e}")
        return None


def _read_git_head(start_dir: Path) -> dict | None:
    """
    The commit and branch of the repository containing 'start_dir', read from its '.git' folder
    without a git executable. Whether the working tree has changes needs git ('dirty': None).
    """
    git_dir = next(
        (d / ".git" for d in (start_dir, *start_dir.parents) if (d / ".git").is_dir()), None
    )
    if git_dir is None:
        return None
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return {"commit": head, "branch": "HEAD", "dirty": None}  # detached HEAD
        ref = head.removeprefix("ref: ")
        ref_file = git_dir / ref
        if ref_file.is_file():
            commit = ref_file.read_text().strip()
        else:  # packed refs: lines of '<commit> <ref>'
            packed = (git_dir / "packed-refs").read_text().splitlines()
            commit = next(line.split()[0] for line in packed if line.endswith(f" {ref}"))
        return {"commit": commit, "branch": ref.removeprefix("refs/heads/"), "dirty": None}
    except (OSError, StopIteration):
        return None


def save_final_config(
    cfg: RunConfig,
    output_dir: str | Path | None = None,
    world_size: int | None = None,
    local_world_size: int | None = None,
) -> None:
    """Save the effective config, defaulting to the configured run output directory.

    The git state of the code repository is saved for reproducibility. If 'world_size' is
    given, the distributed training setup and the resulting global train batch size are
    saved alongside (not inside) the run config.
    """
    output_dir = cfg.general.output_dir.get_resolved_path() if output_dir is None else output_dir
    final_config_path = Path(output_dir, FINAL_CONFIG_FILENAME)
    final_config: dict = {"command": " ".join(sys.argv), "git": _get_git_state()}
    if world_size is not None:
        per_device_batch_size = cfg.trainer.per_device_train_batch_size
        grad_accum_steps = cfg.trainer.gradient_accumulation_steps
        final_config["training_setup"] = {
            "world_size": world_size,
            "local_world_size": local_world_size,
            "per_device_train_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "global_train_batch_size": per_device_batch_size * grad_accum_steps * world_size,
        }
    final_config["run"] = cfg.model_dump()
    save_yaml(final_config, final_config_path, sort_keys=False)


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
        init_source = Path(cfg.general.init_from_checkpoint.get_resolved_path())
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
        )

    return CheckpointPlan(model_checkpoints=model_checkpoints)


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
    vlm_model: AbstractVLWrapper,
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
