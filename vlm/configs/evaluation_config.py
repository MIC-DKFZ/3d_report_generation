from typing import Literal
from pydantic import Field, model_validator
from vlm.configs.base import ConfigBase


class EvaluationConfig(ConfigBase):
    """
    Configuration for the evaluation and inference during the callback and after training.
    """

    # 'Callback' parameters
    eval_every_n_epochs: float = Field(1, description="Run our own callback evaluation on the validation set every n epochs.")

    # 'Callback' and 'After Training' parameters
    max_new_tokens: int = Field(1024, description="Maximum number of new tokens to generate.")
    per_device_batch_size: int = Field(32, description="Inference batch size per device.")

    # 'After Training' parameters
    run_inference: bool = Field(True, description="Run inference after training completes.")
    mode: Literal["best", "final"] = Field("best", description="Which checkpoint to load for inference.")
    split: Literal["val", "test"] = Field("val", description="The split to run inference on.")

    run_eval: bool = Field(True, description="Run evaluation after inference (only if 'run_inference' is 'True').")
    num_eval_processes: int | None = Field(None, description="Number of subprocesses to use for evaluation. "
                                                             "If 'None', use half the number of CPUs in the system.")

    callback_n_samples: int | Literal["unique"] | None = Field(
        None,
        description=(
            "Number of samples. Run all if 'None'. For CT-RATE, use 'unique' to select "
            "one deterministic random image version per unique image."
        ),
    )
    calculate_radbert_score: bool = Field(False, description="Additionally calculate the radbert metrics on evaluation.")

    @model_validator(mode="after")
    def check_inference_eval(self):
        if self.run_eval and not self.run_inference:
            raise ValueError("'run_eval' can only be enabled when 'run_inference' is 'True'.")
        return self
