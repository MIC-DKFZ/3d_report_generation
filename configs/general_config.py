from pydantic import Field

from configs.base import ConfigBase


class GeneralConfig(ConfigBase):
    """
    General training and experiment configuration.
    """
    run_name: str | None = Field(None, description="Name of the run. Also applies naming to WandB.")
    train_config: str | None = Field(None, description="Path to config file (YAML or JSON), "
                                               "relative to the 'VLM_CONFIG_DIR' environment variable")
    output_dir: str | None = Field(None, description="Directory to save model outputs.")
    wandb_project: str = Field("mic-rocket-vlm", description="WandB project name.")
    local_files_only: bool = Field(False, description="Use only local files for Hugging Face from_pretrained.")
    run_inference_eval: bool = Field(True, description="Whether to run inference and evaluation after training.")
