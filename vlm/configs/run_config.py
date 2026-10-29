import argparse
import os
from pathlib import Path
from typing import Any
from pydantic import Field, model_validator
import torch.distributed as dist
from vlm.configs.base import ConfigBase
from vlm.configs.paths import ConfigPath
from vlm.configs.data_config import DataConfig
from vlm.configs.evaluation_config import EvaluationConfig
from vlm.configs.general_config import GeneralConfig
from vlm.configs.model.model_config import ModelConfig
from vlm.configs.scheduler.scheduler_config import SchedulerClassConfig
from vlm.configs.trainer_config import TrainerConfig
from vlm.configs.cli import add_config_cli_args, add_new_class_kwarg_cli_args
from vlm.configs.merging import merge_overrides, recursive_dict_merge, unflatten_dict
from vlm.utils.ddp import is_main_process
from vlm.utils.run_naming import add_datetime_tag, get_random_uid
from vlm.utils.classes import setattr_nested
from vlm.utils.serialization import load_config_file
from vlm.utils.paths import VLM_CONFIG_DIR
from vlm.data.types import VisionInputType
from vlm.utils import logger


class RunConfig(ConfigBase):
    """
    Aggregates all sub-configurations for a training run.

    Currently, the following configurations are held by the RunConfig class:
            - general: general info and configurations
            - model: model configs, trainable parameters
            - data: dataset, images/embeddings, data augs, number of samples
            - trainer: Trainer hyperparameters
            - eval: evaluation parameters
    """
    base_configs: list[str] | None = Field(
        None,
        description="Path(s) to base config(s) to apply first, in left-to-right order."
    )
    general: GeneralConfig
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig
    scheduler: SchedulerClassConfig
    eval: EvaluationConfig

    @model_validator(mode="after")
    def check_eval_dataset_json(self):
        if self.eval.run_eval:
            if self.data.test_dataset_json_path is None:
                raise ValueError("The 'dataset_json' has to be specified if 'run_eval=True'")
        return self

    @model_validator(mode="after")
    def validate_trainable_params_for_text_only_input(self):
        if (
                self.data.text_only
                and getattr(self.model.trainable_params.language_model, "mode", None) == "frozen"
                and getattr(self.model.trainable_params.lm_head, "mode", None) == "frozen"
        ):
            raise ValueError(
                "When 'data.text_only' is enabled, the language model must have trainable parameters. "
                "Otherwise, there are no gradient paths for optimization."
            )
        return self

    @model_validator(mode="after")
    def validate_sharded_vision_input_requires_local_staging(self):
        if self.data.vision_input_is_sharded and not self.general.stage_vision_input_to_local:
            raise ValueError(
                "'data.vision_input_is_sharded=True' requires "
                "'general.stage_vision_input_to_local=True'. Sharded vision input is only "
                "handled by the local staging step; the dataset loader itself expects "
                "normal per-sample files."
            )
        return self

    @model_validator(mode="after")
    def validate_early_stopping_specific_args(self):
        if not self.general.do_early_stopping:
            return self

        if self.data.val_dataset_json_path is None:
            raise ValueError(
                "'general.do_early_stopping=True' requires 'data.val_dataset_json_path' to be set."
            )
        if self.trainer.early_stopping_patience is None:
            raise ValueError(
                "'general.do_early_stopping=True' requires 'trainer.early_stopping_patience' to be set."
            )
        if self.trainer.metric_for_best_model is None:
            raise ValueError(
                "'general.do_early_stopping=True' requires 'trainer.metric_for_best_model' to be set."
            )
        if self.trainer.eval_strategy == "no":
            raise ValueError(
                "'general.do_early_stopping=True' requires 'trainer.eval_strategy' to be 'steps' or 'epoch'."
            )
        if (
            self.trainer.metric_for_best_model in {"eval_classifier_f1", "classifier_f1"}
            and not self.eval.calculate_classifier_score
        ):
            raise ValueError(
                "Early stopping is configured for 'eval_classifier_f1', but 'eval.calculate_classifier_score' is disabled."
            )

        return self

    @model_validator(mode="after")
    def resolve_model_id(self):
        if self.model.model_id:
            return self

        # Resolve model_id if not provided
        from vlm.registries.vlm_registry import VLM_REGISTRY

        if self.model.model_id is None and not self.model.model_type in VLM_REGISTRY:
            raise ValueError(f"Unknown model type: {self.model.model_type}")
        self.model.model_id = ConfigPath(VLM_REGISTRY[self.model.model_type].model_id)

        return self

    @model_validator(mode="after")
    def resolve_run_name(self):
        if self.general.run_name:
            return self

        if is_main_process():
            job_id = os.getenv("SLURM_JOB_ID")
            uid = job_id if job_id else get_random_uid(length=8)
            config_name, _ = os.path.splitext(os.path.basename(self.general.train_config))
            if self.general.run_name_suffix:
                config_name = f"{config_name}__{self.general.run_name_suffix}"
            run_name = [f"{add_datetime_tag(config_name)}__{uid}"]
        else:
            run_name = [None]

        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(run_name, src=0)

            if run_name[0] is None:
                raise RuntimeError(
                    "'run_name' is 'None' after distributed broadcast. "
                    "This indicates a synchronization or initialization error in 'torch.distributed'."
                )
        self.general.run_name = run_name[0]

        return self

    @model_validator(mode="after")
    def resolve_output_dir(self):
        if self.general.output_dir:
            return self

        self.general.output_dir = ConfigPath(
            os.path.join(
                "${VLM_OUTPUT_DIR}",
                self.data.dataset_name,
                os.path.basename(self.data.vision_input_path.get_resolved_path()),
                os.path.dirname(os.path.relpath(self.general.train_config, VLM_CONFIG_DIR)),
                self.general.run_name
            )
        )

        return self

    def resolve_dataloader_workers(self, is_distributed: bool, local_world_size: int) -> None:
        """Calculate unset dataloader workers from available CPUs, capped at four per device."""
        if self.trainer.dataloader_num_workers is not None:
            return

        try:
            num_cores = len(os.sched_getaffinity(0)) or 1
        except (AttributeError, OSError):
            num_cores = os.cpu_count() or 1
        self.trainer.dataloader_num_workers = (
            min(8, max(1, (num_cores // 4) // local_world_size)) if is_distributed else 2
        )

    def get_vision_embed_dim(self) -> int:
        """
        Resolve the vision embedding dimension based on the input source.

        When the vision input type is 'image', the embedding dimension is determined
        by the vision encoder configuration (cfg.model.vision_encoder.embed_dim).
        When precomputed image embeddings are used ('embedding'), the embedding dimension is
        taken from the data configuration (cfg.data.embed_dim).
        """
        if self.data.vision_input_type == VisionInputType.IMAGE:
            return self.model.vision_encoder.embed_dim
        elif self.data.vision_input_type == VisionInputType.EMBEDDING:
            return self.data.embed_dim
        else:
            raise ValueError(f"Unsupported vision input type: {self.cfg.model.vision_input_type}")

    def get_vision_embed_size(self) -> tuple[int, int, int]:
        """
        Resolve the vision embedding size (spatial dims) based on the input source.
        """
        if self.data.vision_input_type == VisionInputType.IMAGE:
            patch_size = self.model.vision_encoder.patch_size
            downsample_factor = self.model.vision_encoder.downsample_factor
            return tuple(p // d for p, d in zip(patch_size, downsample_factor))  # type: ignore[arg-type]
        elif self.data.vision_input_type == VisionInputType.EMBEDDING:
            return self.data.embed_size
        else:
            raise ValueError(f"Unsupported vision input type: {self.cfg.model.vision_input_type}")

    def _maybe_reset_vision_related_specific_args(self):
        """
        When using embeddings or using text-only inputs, reset vision-related-specific arguments:
        - set object references to None
        - set preprocessing flags to False
        """
        embedding_input = self.data.vision_input_type == VisionInputType.EMBEDDING
        text_only = self.data.text_only

        if not (embedding_input or text_only):
            return

        _dic = {
            None: set(),
            False: set(),
        }
        _str = None
        if embedding_input:
            _str = "EMBEDDING"
            _dic[None].update(["model.vision_encoder", "model.trainable_params.vision_encoder"])
            _dic[False].update(["data.do_resize", "data.do_transforms"])
        if text_only:
            _str = "TEXT-ONLY"
            _dic[None].update(["model.vision_encoder", "model.trainable_params.vision_encoder",
                               "model.adapter", "model.trainable_params.adapter"])
            _dic[False].update(["data.do_resize", "data.do_transforms"])

        logger.info_main(f"[Vision Input Type] Resetting vision-encoder-specific arguments for {_str} input mode:")

        for _val, _modules in _dic.items():
            for _module in sorted(_modules):
                setattr_nested(self, _module, _val)
                logger.info_main(f"  > 'cfg.{_module}' set to {_val}")

    def _maybe_reset_validation_specific_args(self):
        """
        When doing no validation ('val_dataset_json_path=None'), reset validation-specific arguments.
        This is especially important for Trainer args, which would otherwise raise errors.
        """
        if self.data.val_dataset_json_path is None:
            logger.info_main("[No Validation] Resetting validation-specific arguments "
                             "as no validation dataset JSON was provided:")

            _lis = [
                (["trainer.eval_steps", "trainer.metric_for_best_model"], None),
                (["trainer.eval_strategy"], "no"),
                (["general.do_early_stopping"], False),
            ]
            for _modules, _val in _lis:
                for _module in _modules:
                    setattr_nested(self, _module, _val)
                    if is_main_process():
                        logger.info(f"  > 'cfg.{_module}' set to {_val}")

    def _maybe_reset_early_stopping_specific_args(self):
        """
        When early stopping is disabled, reset early-stopping-specific arguments.
        This keeps inherited base-config values from enabling the callback implicitly.
        """
        if self.general.do_early_stopping:
            return

        logger.info_main("[Early Stopping] Resetting early-stopping-specific arguments:")

        _lis = [
            (["trainer.early_stopping_patience"], None),
            (["trainer.early_stopping_threshold"], 0.0),
        ]
        for _modules, _val in _lis:
            for _module in _modules:
                setattr_nested(self, _module, _val)
                if is_main_process():
                    logger.info(f"  > 'cfg.{_module}' set to {_val}")

    def apply_conditional_resets(self) -> None:
        self._maybe_reset_vision_related_specific_args()
        self._maybe_reset_validation_specific_args()
        self._maybe_reset_early_stopping_specific_args()

    @classmethod
    def validate_with_cli_args(cls, cfg_dict: dict[str, Any], cli_args: list[str]) -> "RunConfig":
        """
        Validate a config dictionary with dotted CLI overrides applied.

        The dictionary is validated once to register the override options, and again with the
        overrides, so that derived fields such as the run name and output directory reflect them.
        """
        cfg = cls.model_validate(cfg_dict)
        overrides = cfg.parse_cli_args(cli_args)
        if not overrides:
            return cfg
        return cls.model_validate(merge_overrides(cfg_dict, overrides, cfg.model_dump()))

    def parse_cli_args(self, cli_args: list[str]) -> dict[str, Any]:
        """Parse dotted CLI overrides into a nested dictionary, rejecting unknown arguments."""
        if not cli_args:
            return {}

        parser = argparse.ArgumentParser(allow_abbrev=False)
        dynamic_kwargs_paths = set()
        for name in type(self).model_fields:
            subconfig = getattr(self, name)
            if isinstance(subconfig, ConfigBase):
                dynamic_kwargs_paths.update(add_config_cli_args(parser, subconfig, parent_key=name))

        add_new_class_kwarg_cli_args(parser, cli_args, dynamic_kwargs_paths)

        return unflatten_dict(vars(parser.parse_args(cli_args)))

    def merge_subconfigs_with_cli_args(self, cli_args: list[str]) -> None:
        """
        Apply dotted CLI overrides to this validated config, rejecting unknown arguments.

        Derived fields such as the run name and output directory keep their values; use
        'validate_with_cli_args' for configs whose derived fields should reflect the overrides.
        """
        for name, overrides in self.parse_cli_args(cli_args).items():
            subconfig = getattr(self, name)
            merged = recursive_dict_merge(subconfig.model_dump(), overrides)
            setattr(self, name, type(subconfig).model_validate(merged))

    @staticmethod
    def apply_base_configs(cfg_dict: dict[str, Any], config_dir: str | Path, seen: list[str] | None = None) -> dict:
        """
        Recursively applies base configurations defined by the 'base_configs' key in 'cfg_dict'.

        Base configs are merged in reverse order so that earlier entries override later ones.
        Values in 'cfg_dict' itself take final precedence over all base configurations.

        Raises:
            RecursionError: If a cyclic base config reference is detected.
        """
        base_configs: list[str] | None = cfg_dict.get("base_configs")
        if not base_configs:
            return cfg_dict

        seen = seen or []
        merged_bases = {}
        for base_config in reversed(base_configs):
            if base_config in seen:
                raise RecursionError(
                    f"Cyclic base config reference: {base_config} found in {seen}"
                )
            base_cfg_dict = load_config_file(Path(config_dir, base_config))
            base_cfg_dict = RunConfig.apply_base_configs(
                base_cfg_dict, config_dir, seen=seen + [base_config]
            )
            merged_bases = recursive_dict_merge(merged_bases, base_cfg_dict)

        return recursive_dict_merge(merged_bases, cfg_dict)
