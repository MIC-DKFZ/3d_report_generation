from pydantic import Field

from vlm.configs.base import ConfigBase
from vlm.configs.paths import ConfigPath


class GeneralConfig(ConfigBase):
    """
    General training and experiment configuration.
    """
    run_name: str | None = Field(None, description="Name of the run. Also applies naming to WandB.")
    run_name_suffix: str | None = Field(
        None,
        description="Suffix after the config name in generated run names, e.g. to tell apart runs "
        "of one config on different dataset JSONs. Ignored when 'run_name' is set.",
    )
    train_config: str | None = Field(None, description="Path to config file (YAML or JSON), "
                                               "relative to the 'VLM_CONFIG_DIR' environment variable")
    seed: int | None = Field(42, description="Random seed for reproducibility.")
    init_from_checkpoint: ConfigPath | None = Field(
        None,
        description=(
            "Initialize model weights from a checkpoint directory path, or from the final "
            "checkpoint of a run identified by its run directory name under 'VLM_OUTPUT_DIR'."
        ),
    )
    output_dir: ConfigPath | None = Field(None, description="Directory to save model outputs.")
    wandb_project: str | None = Field(None, description="WandB project name.")
    wandb_entity: str | None = Field(None, description="WandB account or team name.")
    do_early_stopping: bool = Field(False, description="Enable Trainer early stopping callback.")
    local_files_only: bool = Field(False, description="Use only local files for Hugging Face 'from_pretrained'.")
    stage_vision_input_to_local: bool = Field(
        False,
        description="Copy or extract the configured vision input directory to node-local storage before training.",
    )
