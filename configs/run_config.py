import os
from pathlib import Path
from typing import Any
from pydantic import Field, model_validator
import torch.distributed as dist
from configs.base import ConfigBase
from configs.data_config import DataConfig
from configs.evaluation_config import EvaluationConfig
from configs.general_config import GeneralConfig
from configs.model.model_config import ModelConfig
from configs.trainer_config import SFTTrainerConfig
from utils.helper import load_config_file, recursive_dict_merge, add_datetime_tag, setattr_nested, is_main_process, \
    get_random_uid
from utils.paths import VLM_OUTPUT_DIR, VLM_CONFIG_DIR
from utils.vision_input_type import VisionInputType
from utils import logger


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
    eval: EvaluationConfig

    @model_validator(mode="after")
    def set_default_eval_dataset_json_path(self):
        # fall back to the val dataset JSON from data config if none is provided
        if self.eval.split == "val" and self.eval.dataset_json_path is None:
            self.eval.dataset_json_path = self.data.val_dataset_json_path
        return self

    @model_validator(mode="after")
    def check_eval_dataset_json(self):
        if self.eval.run_eval:
            # 'dataset_json_path' can only be None if split='test' and no dataset JSON was provided.
            # RunConfig.set_default_eval_dataset_json_path() assigns the validation dataset JSON
            # when split='val' and no path is specified.
            if self.eval.split == "test" and self.eval.dataset_json_path is None:
                raise ValueError("The 'dataset_json' has to be specified if 'split=test'")
            if not Path(self.eval.dataset_json_path).is_file():
                raise FileNotFoundError(f"The specified dataset JSON file does not exist: {self.eval.dataset_json_path}")
        return self

    @model_validator(mode="after")
    def check_trainable_params_for_text_only_input(self):
        if (
                self.data.text_only
                and self.model.trainable_params.language_model.mode == "frozen"
                and self.model.trainable_params.lm_head.mode == "frozen"
        ):
            raise ValueError(
                "When 'data.text_only' is enabled, the language model must have trainable parameters. "
                "Otherwise, there are no gradient paths for optimization."
            )
        return self

    def maybe_reset_vision_encoder_specific_args(self):
        """
        When using embeddings, reset vision-encoder-specific arguments:
        - set object references to None
        - set preprocessing flags to False
        """
        if self.data.vision_input_type == VisionInputType.EMBEDDING:
            if is_main_process():
                logger.info("[VisionInputType] Resetting vision-encoder-specific arguments for EMBEDDING input mode:")

            _lis = [
                (["model.vision_encoder", "model.trainable_params.vision_encoder"], None),
                (["data.do_resize", "data.do_transforms"], False),
            ]
            for _modules, _val in _lis:
                for _module in _modules:
                    setattr_nested(self, _module, _val)
                    if is_main_process():
                        logger.info(f"  > cfg.{_module} set to {_val}")

    def resolve_run_name(self) -> None:
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

    def resolve_output_dir(self) -> None:
        self.general.output_dir = os.path.join(
            VLM_OUTPUT_DIR,
            self.data.dataset_name,
            os.path.basename(self.data.vision_input_path),
            os.path.dirname(os.path.relpath(self.general.train_config, VLM_CONFIG_DIR)),
            self.general.run_name
        )

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