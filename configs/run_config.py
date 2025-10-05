import os
from pathlib import Path
from typing import Any
from pydantic import Field
from configs.base import ConfigBase
from configs.data_config import DataConfig
from configs.general_config import GeneralConfig
from configs.model.model_config import ModelConfig
from configs.trainer_config import SFTTrainerConfig
from utils.helper import load_config_file, recursive_dict_merge, add_datetime_tag, setattr_nested
from utils.vision_input_type import VisionInputType


class RunConfig(ConfigBase):
    """
    Aggregates all sub-configurations for a training run.

    Currently, the following configurations are held by the RunConfig class:
            - general: general info and configurations
            - model: model configs, trainable parameters
            - data: dataset, images/embeddings, data augs, number of samples
            - trainer: SFTTrainer hyperparameters
    """
    base_configs: list[str] | None = Field(
        None,
        description="Path(s) to base config(s) to apply first, in left-to-right order."
    )
    general: GeneralConfig
    model: ModelConfig
    data: DataConfig
    trainer: SFTTrainerConfig

    def maybe_reset_vision_encoder_specific_args(self, logger=None):
        """
        When using embeddings, reset vision-encoder-specific arguments:
        - set object references to None
        - set preprocessing flags to False
        """
        if self.data.vision_input_type == VisionInputType.EMBEDDING:
            if logger:
                logger.info("[VisionInputType] Resetting vision-encoder-specific arguments for EMBEDDING input mode:")

            _lis = [
                (["model.vision_encoder", "model.trainable_params.vision_encoder", "data.embedding_constant"], None),
                (["data.do_resize", "data.do_transforms"], False),
            ]
            for _modules, _val in _lis:
                for _module in _modules:
                    setattr_nested(self, _module, _val)
                    if logger:
                        logger.info(f"  > cfg.{_module} set to {_val}")

    def resolve_run_name(self) -> None:
        config_name, _ = os.path.splitext(os.path.basename(self.general.train_config))
        self.general.run_name = add_datetime_tag(config_name)
        
    def resolve_output_dir(self, output_dir: str | Path) -> None:
        self.general.output_dir = os.path.join(
            output_dir,
            self.data.dataset_name,
            os.path.basename(self.data.vision_input_path),
            # os.path.dirname(os.path.relpath(self.general.train_config, config_dir)),
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