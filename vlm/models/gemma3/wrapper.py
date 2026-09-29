from vlm.models.gemma3.modeling_gemma3_text_vl import Gemma3TextVLForConditionalGeneration
from vlm.models.text_backbone_vl_wrapper import TextBackboneVLWrapper


GEMMA3_VISION_PLACEHOLDER_TOKEN = "<image_soft_token>"


class Gemma3TextVLWrapper(TextBackboneVLWrapper):
    model_class = Gemma3TextVLForConditionalGeneration
    model_name = "gemma3"
    vision_placeholder_token = GEMMA3_VISION_PLACEHOLDER_TOKEN
    vision_token_id_attr = "image_token_id"
    vision_content_types = ("image", "video")
    extra_forbidden_name_patterns_for_wd = ("lora", "embed_tokens", "lm_head")

    def _configure_tokenizer(self) -> None:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
