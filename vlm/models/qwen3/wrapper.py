from vlm.models.qwen3.modeling_qwen3_text_vl import Qwen3TextVLForConditionalGeneration
from vlm.models.text_backbone_vl_wrapper import TextBackboneVLWrapper


class Qwen3TextVLWrapper(TextBackboneVLWrapper):
    model_class = Qwen3TextVLForConditionalGeneration
    model_name = "qwen3"
    vision_placeholder_token = "<|video_pad|>"
    extra_forbidden_name_patterns_for_wd = ("lora", "embed_tokens", "lm_head")
    # Training renders an empty think block before every response, so the generation prompt must
    # pre-fill it too instead of leaving the model to start thinking (the template's default).
    chat_template_kwargs = {"enable_thinking": False}

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool) -> str:
        text = super()._apply_chat_template(messages, add_generation_prompt)
        if not add_generation_prompt:
            text = text.removesuffix("\n")
        return text
