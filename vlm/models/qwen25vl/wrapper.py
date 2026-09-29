import torch
import torch.nn.functional as F
from vlm.models.qwen25vl.processing_3d import QwenVLProcessor3D
from vlm.models.qwen25vl.modeling_qwen25_vl import Qwen25VLForConditionalGeneration
from vlm.utils.filter_batch import extract_raw_samples_from_batch
from vlm.utils.paths import get_hf_hub_cache_dir

from vlm.models.abstract_vl_wrapper import AbstractVLWrapper


class Qwen25VLWrapper(AbstractVLWrapper):
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

        self.model = Qwen25VLForConditionalGeneration.from_pretrained(
            self.model_cfg.model_id,
            torch_dtype=torch.bfloat16
        )

        self._setup_vision_components()
        if self.model_cfg.vision_encoder is not None:
            self.downsample_factor = self.model_cfg.vision_encoder.downsample_factor

        self.processor = QwenVLProcessor3D.from_pretrained(
            self.model_cfg.model_id,
            cache_dir=cache_dir
        )
        self.tokenizer = self.processor.tokenizer

        return self.model

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

        batch = self.processor(
            text=texts,
            num_vision_tokens=num_vision_tokens,
            do_truncation=self.do_truncation,
            truncate_after=self.truncate_after,
            truncate_at=self.truncate_at,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )

        batch_prompt_only = self.processor(
            text=texts_prompt_only,
            num_vision_tokens=num_vision_tokens,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        if vision_inputs is not None:
            if isinstance(self.processor, QwenVLProcessor3D):
                image_tokens = [151652, 151653, 151655]
            else:
                image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]

            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

        # Only keep the model's final assistant response as the prediction target.
        # Everything before and including the last "<|im_start|>assistant\n"
        # as well as after the last "<|im_end|> (e.g. pad tokens) is ignored (-100).
        im_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        for i in range(labels.shape[0]):
            last_im_start_pos = (labels[i] == im_start_token_id).nonzero(as_tuple=True)[0][-1]
            last_im_end_pos = (labels[i] == im_end_token_id).nonzero(as_tuple=True)[0][-1]
            labels[i, :last_im_start_pos + 3] = -100
            labels[i, last_im_end_pos + 1:] = -100

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
