from typing import Optional
from transformers.modeling_utils import unwrap_model
from trl import SFTTrainer
from utils.helper import get_trainable_state_dict


class VLMSFTTrainer(SFTTrainer):

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