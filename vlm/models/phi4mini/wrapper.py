from vlm.models.phi4mini.modeling_phi4mini_vl import Phi4MiniVLForConditionalGeneration
from vlm.models.text_backbone_vl_wrapper import TextBackboneVLWrapper


class Phi4MiniVLWrapper(TextBackboneVLWrapper):
    model_class = Phi4MiniVLForConditionalGeneration
    model_name = "phi4mini"
    vision_placeholder_token = "<|endofprompt|>"

    def _configure_tokenizer(self) -> None:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
