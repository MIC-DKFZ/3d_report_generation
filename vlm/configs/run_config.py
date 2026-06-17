import os
from pathlib import Path
from typing import Any
from pydantic import Field, model_validator
import torch.distributed as dist
from vlm.configs.base import ConfigBase
from vlm.configs.data_config import DataConfig
from vlm.configs.evaluation_config import EvaluationConfig
from vlm.configs.general_config import GeneralConfig
from vlm.configs.model.model_config import ModelConfig
from vlm.configs.scheduler.scheduler_config import SchedulerClassConfig
from vlm.configs.trainer_config import SFTTrainerConfig
from vlm.utils.helper import load_config_file, recursive_dict_merge, add_datetime_tag, setattr_nested, is_main_process, \
    get_random_uid, merge_config_with_cli_args
from vlm.utils.paths import VLM_OUTPUT_DIR, VLM_CONFIG_DIR
from vlm.utils.vision_input_type import VisionInputType
from vlm.utils import logger


class RunConfig(ConfigBase):
    """
    Aggregates all sub-configurations for a training run.

    Currently, the following configurations are held by the RunConfig class:
            - general: general info and configurations
            - model: model configs, trainable parameters
            - data: dataset, images/embeddings, data augs, number of samples
            - trainer: SFTTrainer hyperparameters
            - eval: evaluation parameters
    """
    base_configs: list[str] | None = Field(
        None,
        description="Path(s) to base config(s) to apply first, in left-to-right order."
    )
    general: GeneralConfig
    model: ModelConfig
    data: DataConfig
    trainer: SFTTrainerConfig
    scheduler: SchedulerClassConfig
    eval: EvaluationConfig

    @model_validator(mode="after")
    def check_eval_dataset_json(self):
        if self.eval.run_eval:
            if self.eval.test_dataset_json_path is None:
                raise ValueError("The 'dataset_json' has to be specified if 'run_eval=True'")
            if not Path(self.eval.test_dataset_json_path).is_file():
                raise FileNotFoundError(f"The specified dataset JSON file does not exist: {self.eval.test_dataset_json_path}")
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
    def validate_early_stopping_specific_args(self):
        if not self.general.do_early_stopping:
            return self

        if self.data.val_dataset_json_path is None:
            raise ValueError(
                "general.do_early_stopping=True requires data.val_dataset_json_path to be set."
            )
        if self.trainer.early_stopping_patience is None:
            raise ValueError(
                "general.do_early_stopping=True requires trainer.early_stopping_patience to be set."
            )
        if self.trainer.metric_for_best_model is None:
            raise ValueError(
                "general.do_early_stopping=True requires trainer.metric_for_best_model to be set."
            )
        if self.trainer.eval_strategy == "no":
            raise ValueError(
                "general.do_early_stopping=True requires trainer.eval_strategy to be 'steps' or 'epoch'."
            )
        if (
            self.trainer.metric_for_best_model in {"eval_radbert_f1", "radbert_f1"}
            and not self.eval.calculate_radbert_score
        ):
            raise ValueError(
                "Early stopping is configured for 'eval_radbert_f1', but eval.calculate_radbert_score is disabled."
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
        self.model.model_id = self.model.model_id or VLM_REGISTRY[self.model.model_type].model_id

        return self

    @model_validator(mode="after")
    def resolve_run_name(self):
        if self.general.run_name:
            return self

        if is_main_process():
            job_id = os.getenv("SLURM_JOB_ID")
            uid = job_id if job_id else get_random_uid(length=8)
            config_name, _ = os.path.splitext(os.path.basename(self.general.train_config))
            run_name = [f"{add_datetime_tag(config_name)}__{uid}"]
        else:
            run_name = [None]

        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(run_name, src=0)

            if run_name[0] is None:
                raise RuntimeError(
                    "run_name is None after distributed broadcast. "
                    "This indicates a synchronization or initialization error in torch.distributed."
                )
        self.general.run_name = run_name[0]

        return self

    @model_validator(mode="after")
    def resolve_output_dir(self):
        if self.general.output_dir:
            return self

        self.general.output_dir = os.path.join(  # type: ignore[arg-type]
            VLM_OUTPUT_DIR,
            self.data.dataset_name,
            os.path.basename(self.data.vision_input_path),
            os.path.dirname(os.path.relpath(self.general.train_config, VLM_CONFIG_DIR)),
            self.general.run_name
        )

        return self

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
                logger.info_main(f"  > cfg.{_module} set to {_val}")

    def _maybe_reset_validation_specific_args(self):
        """
        When doing no validation (val_dataset_json_path=None), reset validation-specific arguments.
        This is especially important for SFTTrainer args, which would otherwise raise errors.
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
                        logger.info(f"  > cfg.{_module} set to {_val}")

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
                    logger.info(f"  > cfg.{_module} set to {_val}")

    def apply_conditional_resets(self) -> None:
        self._maybe_reset_vision_related_specific_args()
        self._maybe_reset_validation_specific_args()
        self._maybe_reset_early_stopping_specific_args()

    def merge_subconfigs_with_cli_args(self):
        """ Apply CLI argument overrides to configs """
        for subconfig_name, field in self.model_fields.items():
            subconfig = getattr(self, subconfig_name)
            if isinstance(subconfig, ConfigBase):
                setattr(self, subconfig_name, merge_config_with_cli_args(subconfig, parent_key=subconfig_name))

    @staticmethod
    def apply_base_configs(cfg_dict: dict[str, Any], config_dir: str | Path, seen: list[str] | None = None) -> dict:
        """
        Recursively applies base configurations defined in 'cfg_dict["base_configs"]'.

        Base configs are merged in reverse order so that earlier entries override later ones.
        Values in 'cfg_dict' itself take final precedence over all base configurations.

        Raises:
            RecursionError: If a cyclic base config reference is detected.
        """
        base_configs: list[str] | None = cfg_dict.get("base_configs")
        if not base_configs:
            return cfg_dict

        seen = seen.copy() if seen else []

        new_cfg_dict = {}
        for base_config in reversed(base_configs):
            if base_config in seen:
                raise RecursionError(
                    f"Cyclic base config reference: {base_config} found in {seen}"
                )
            seen.append(base_config)

            base_cfg_dict = load_config_file(Path(config_dir, base_config))
            base_cfg_dict = RunConfig.apply_base_configs(base_cfg_dict, config_dir, seen=seen)

            # 'base_cfg_dict' overrides 'new_cfg_dict'
            new_cfg_dict = recursive_dict_merge(new_cfg_dict, base_cfg_dict)

            seen.pop()

        new_cfg_dict = recursive_dict_merge(new_cfg_dict, cfg_dict)

        return new_cfg_dict