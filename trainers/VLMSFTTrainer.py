from typing import Optional
from trl import SFTTrainer


class VLMSFTTrainer(SFTTrainer):

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Only save trainable parameters with 'requires_grad=True'.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        trainable_names = {name for name, p in self.model.named_parameters() if p.requires_grad}

        state_dict = {k: v for k, v in self.model.state_dict().items() if k in trainable_names}

        if self.args.should_save:
            self._save(output_dir, state_dict)