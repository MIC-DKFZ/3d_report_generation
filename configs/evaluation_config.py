from typing import Literal
from pydantic import Field, model_validator
from configs.base import ConfigBase


class EvaluationConfig(ConfigBase):
    """
    Configuration for the evaluation and inference during the callback and after training.
    """

    # 'Callback' parameters
    eval_every_n_epochs: float = Field(0.25, description="Run our own callback evaluation on the validation set every n epochs.")

    # 'Callback' and 'After Training' parameters
    max_new_tokens: int = Field(640, description="Maximum number of new tokens to generate.")
    per_device_batch_size: int = Field(12, description="Inference batch size per device.")

    # 'After Training' parameters
    run_inference: bool = Field(False, description="Run inference after training completes.")
    dataset_json_path: str | None = Field(
        None,
        description=(
            "Optional path to the dataset JSON file. "
            "If not provided and split='val', the validation dataset JSON from the data config is used. "
            "If not provided and split='test', an error is raised. "
            "Specify this path explicitly when running validation in a different environment, "
            "or when performing final testing, where the test set JSON must be provided."
        ),
    )
    mode: Literal["best_metric", "best", "final"] = Field("final", description="Which checkpoint to load for inference.")
    split: Literal["val", "test"] = Field("val", description="The split to run inference on.")
    n_samples: int | None = Field(None, description="Number of samples. Run all if 'None'.")
    num_dataloader_workers: int | None = Field(4, description="Number of subprocesses per device to use for data loading during inference.")

    run_eval: bool = Field(False, description="Run evaluation after inference (only if 'run_inference' is True).")
    num_eval_processes: int | None = Field(None, description="Number of subprocesses to use for evaluation. "
                                                             "If None, use half the number of CPUs in the system.")

    @model_validator(mode="after")
    def check_inference_eval(self):
        if self.run_eval and not self.run_inference:
            raise ValueError("'run_eval' can only be enabled when 'run_inference' is True.")
        return self
