from vlm.models.qwen35.modeling_qwen35 import Qwen35VLForConditionalGeneration
from vlm.models.qwen3vl.wrapper import Qwen3VLWrapper


class Qwen35VLWrapper(Qwen3VLWrapper):
    """Qwen3.5 shares the Qwen3-VL processor, chat template, and adapter integration."""

    model_class = Qwen35VLForConditionalGeneration
