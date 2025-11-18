import torch.nn as nn
from typing import Optional
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import get_parameter_names
from trl import SFTTrainer
from vlm.utils.helper import get_trainable_state_dict


class VLMSFTTrainer(SFTTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This flag is typically True by default.
        # For some reason, this can result in the training loss not being normalized by the batch size.
        self.model_accepts_loss_kwargs = False

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Only save trainable parameters with 'requires_grad=True'.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        model_to_save = unwrap_model(self.model)
        trainable_state_dict = get_trainable_state_dict(model_to_save)

        if self.args.should_save:
            self._save(output_dir, trainable_state_dict)

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden_name_patterns = [
            r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)",
            # Add LoRA layers and the token embedding layer (which is also the LM head) to forbidden name patterns.
            "lora", "embed_tokens", "lm_head"
        ]
        decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters