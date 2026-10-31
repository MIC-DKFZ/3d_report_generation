from typing import Literal
from pydantic import Field, model_validator
from vlm.configs.base import ConfigBase


class EvaluationConfig(ConfigBase):
    """
    Configuration for the evaluation and inference during the callback and after training.
    """

    eval_every_n_epochs: int = Field(
        1,
        ge=1,
        description=(
            "Evaluate every N whole epochs when trainer.eval_strategy='epoch'. "
            "The final epoch is always evaluated."
        ),
    )

    # 'Callback' and 'After Training' parameters
    max_new_tokens: int = Field(1024, description="Maximum number of new tokens to generate.")
    per_device_batch_size: int = Field(32, description="Inference batch size per device.")

    # Decoding of the callback and of inference. Greedy by default, so all models decode alike: their
    # generation configs differ, e.g. Qwen3-VL samples (temperature 0.7, top-p 0.8, top-k 20) while
    # Qwen3.5 ships none and decodes greedily. 'None' keeps the model's own setting.
    do_sample: bool | None = Field(
        False, description="Sample instead of decoding greedily. 'None' keeps the model's setting."
    )
    temperature: float | None = Field(None, gt=0, description="Sampling temperature.")
    top_p: float | None = Field(None, gt=0, le=1, description="Nucleus sampling probability mass.")
    top_k: int | None = Field(None, ge=0, description="Number of most likely tokens to sample from.")

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
    calculate_classifier_score: bool = Field(
        False,
        description="Calculate dataset-specific classifier metrics during callback evaluation.",
    )

    def generation_kwargs(self) -> dict:
        """Decoding arguments for 'generate' that override the model's generation config."""
        settings = {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        return {name: value for name, value in settings.items() if value is not None}

    @model_validator(mode="after")
    def check_inference_eval(self):
        if self.run_eval and not self.run_inference:
            raise ValueError("'run_eval' can only be enabled when 'run_inference' is 'True'.")
        return self
