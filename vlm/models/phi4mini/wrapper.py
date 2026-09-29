from vlm.models.phi4mini.modeling_phi4mini_vl import Phi4MiniVLForConditionalGeneration
from vlm.models.text_backbone_vl_wrapper import TextBackboneVLWrapper


PHI_VISION_PLACEHOLDER_TOKEN = "<|endofprompt|>"


class Phi4MiniVLWrapper(TextBackboneVLWrapper):
    model_class = Phi4MiniVLForConditionalGeneration
    model_name = "phi4mini"
    vision_placeholder_token = PHI_VISION_PLACEHOLDER_TOKEN

    def _configure_tokenizer(self) -> None:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _content_to_text(content) -> str:
        if isinstance(content, str):
            return content

        raise TypeError(
            "Phi4Mini expects string chat message content. "
            "Structured multimodal content is currently only supported by Qwen."
        )
