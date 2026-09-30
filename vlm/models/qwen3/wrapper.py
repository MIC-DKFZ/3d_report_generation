from vlm.models.qwen3.modeling_qwen3_text_vl import Qwen3TextVLForConditionalGeneration
from vlm.models.text_backbone_vl_wrapper import TextBackboneVLWrapper


class Qwen3TextVLWrapper(TextBackboneVLWrapper):
    model_class = Qwen3TextVLForConditionalGeneration
    model_name = "qwen3"
    vision_placeholder_token = "<|video_pad|>"
    extra_forbidden_name_patterns_for_wd = ("lora", "embed_tokens", "lm_head")

    @staticmethod
    def _strip_empty_thinking_block(text: str) -> str:
        return text.replace("<think>\n\n</think>\n\n", "")

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool) -> str:
        text = super()._apply_chat_template(messages, add_generation_prompt)
        text = self._strip_empty_thinking_block(text)
        if not add_generation_prompt:
            text = text.removesuffix("\n")
        return text
