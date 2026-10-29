from vlm.models.qwen35.modeling_qwen35 import Qwen35VLForConditionalGeneration
from vlm.models.qwen3vl.wrapper import Qwen3VLWrapper


class Qwen35VLWrapper(Qwen3VLWrapper):
    """Qwen3.5 shares the Qwen3-VL processor, chat template, and adapter integration."""

    model_class = Qwen35VLForConditionalGeneration
    # Training renders an empty think block before every response. The 4B template opens a thinking
    # block in the generation prompt by default (the 2B template does not), so disable it explicitly.
    chat_template_kwargs = {"enable_thinking": False}
