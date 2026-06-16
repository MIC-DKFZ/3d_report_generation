from pydantic import Field

from vlm.configs.base import ConfigBase


class GeneralConfig(ConfigBase):
    """
    General training and experiment configuration.
    """
    run_name: str | None = Field(None, description="Name of the run. Also applies naming to WandB.")
    train_config: str | None = Field(None, description="Path to config file (YAML or JSON), "
                                               "relative to the 'VLM_CONFIG_DIR' environment variable")
    seed: int | None = Field(42, description="Random seed for reproducibility.")
    init_from_checkpoint_dir: str | None = Field(None, description="Directory to load model from.")
    init_optimizer_scheduler_from_checkpoint: bool = Field(
        False,
        description=(
            "If True and `init_from_checkpoint_dir` is set, also initialize optimizer and LR scheduler "
            "state from that checkpoint directory."
        ),
    )
    output_dir: str | None = Field(None, description="Directory to save model outputs.")
    wandb_project: str = Field("mic-rocket-vlm", description="WandB project name.")
    local_files_only: bool = Field(False, description="Use only local files for Hugging Face from_pretrained.")
    stage_vision_input_to_local: bool = Field(
        False,
        description="Copy or extract the configured vision input directory to node-local storage before training.",
    )
