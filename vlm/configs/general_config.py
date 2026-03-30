from typing import Literal
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
    init_from_checkpoint: str | None = Field(None, description="Directory to load model from.")
    output_dir: str | None = Field(None, description="Directory to save model outputs.")
    wandb_project: str = Field("mic-rocket-vlm", description="WandB project name.")
    local_files_only: bool = Field(False, description="Use only local files for Hugging Face from_pretrained.")
    continue_training: bool = Field(False, description="Training will continue from state defined in `output_dir`.")
    continue_training_mode: Literal["best_metric", "best", "final"] = Field("final", description="Which checkpoint to load to continue training.")
