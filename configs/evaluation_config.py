from pydantic import Field
from configs.base import ConfigBase


class EvaluationConfig(ConfigBase):
    """
    Configuration for the evaluation that takes place during the callback function.
    """
    eval_every_n_epochs: float = Field(0.25, description="Run our own evaluation on validation set every n epochs.")
    max_new_tokens: int = Field(640, description="Maximum number of new tokens to generate.")
    batch_size: int = Field(48, description="Inference batch size per device.")
