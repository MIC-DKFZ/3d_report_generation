import torch
from vlm.models.qwen3vl.processing_3d import QwenVLProcessor3D
from vlm.models.qwen3vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from vlm.data.batches import extract_raw_samples_from_batch
from vlm.utils.paths import get_hf_hub_cache_dir

from vlm.models.abstract_vl_wrapper import AbstractVLWrapper


class Qwen3VLWrapper(AbstractVLWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_str = "qwen"
        self.batch_count = 0

        self.forbidden_name_patterns_for_wd.extend(
            # Add LoRA layers and the token embedding layer (which is also the LM head) to forbidden name patterns.
            ["lora", "embed_tokens", "lm_head"]
        )

    def setup_model(self):
        """Initialize model, processor and tokenizer"""

        cache_dir = get_hf_hub_cache_dir()

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_cfg.model_id.get_resolved_path(),
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
            dtype=torch.bfloat16,
        )

        self._setup_vision_components()

        # Uncomment to exactly reproduce RMR experiments; see README reproducibility section.
        # self.model.post_init()

        if self.model_cfg.vision_encoder is not None:
            self.downsample_factor = self.model.model.vision_encoder.downsample_factor

        self.processor = QwenVLProcessor3D.from_pretrained(
            self.model_cfg.model_id.get_resolved_path(),
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        )
        self.tokenizer = self.processor.tokenizer

        return self.model

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return super()._apply_chat_template(
            self._replace_vision_content_type(messages, content_type="video"),
            add_generation_prompt=add_generation_prompt,
        )

    def collate_fn(self, examples):
        """Custom collate function for Qwen model"""
        texts = [
            self._apply_chat_template(example["formatted_qa"], add_generation_prompt=False)
            for example in examples
        ]
        texts_prompt_only = [
            self._apply_chat_template(
                example["formatted_qa_prompt_only"], add_generation_prompt=True
            )
            for example in examples
        ]

        vision_inputs = None
        num_vision_tokens, vision_position_ids, adapter_args = None, None, {}

        if not self.text_only:
            raw_vision_inputs = [
                example["vision_input"]
                for example in examples
            ]

            # Support both single scans and list/tuple batches of scans.
            vision_inputs = []
            for vi in raw_vision_inputs:
                if isinstance(vi, (list, tuple)):
                    vision_inputs.extend(vi)
                else:
                    vision_inputs.append(vi)

            (
                vision_inputs,
                num_vision_tokens,
                vision_position_ids,
                adapter_args,
            ) = self._prepare_vision_inputs(vision_inputs)

        batch_prompt_only = self.processor(
            text=texts_prompt_only,
            num_vision_tokens=num_vision_tokens,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        prompt_lengths = batch_prompt_only["attention_mask"].sum(dim=1).tolist()
        batch = self.processor(
            text=texts,
            num_vision_tokens=num_vision_tokens,
            padding=False,
        )
        batch = self._truncate_and_pad(batch, prompt_lengths)

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        # Only the response contributes to the loss; ignore the complete prompt.
        for i, prompt_length in enumerate(prompt_lengths):
            labels[i, :prompt_length] = -100

        raw_samples = extract_raw_samples_from_batch(examples)
        batch_out = {
            "input_ids": batch["input_ids"],
            "input_ids_prompt_only": batch_prompt_only["input_ids"],
            "attention_mask": batch["attention_mask"],
            "attention_mask_prompt_only": batch_prompt_only["attention_mask"],
            "labels": labels,
            "pixel_values_videos": vision_inputs,
            "batch_count": self.batch_count,
            "raw_samples": raw_samples,
            "vision_position_ids": vision_position_ids,
            "adapter_args": adapter_args
        }
        self.batch_count += 1
        return batch_out
