import os
import torch
import numpy as np
from vlm.models.qwen25vl.processing_3d import QwenVLProcessor3D
from vlm.models.qwen25vl.Qwen3D import QwenVLForConditionalGeneration3D, QwenVLModel3D
from vlm.utils.filter_batch import extract_raw_samples_from_batch

from vlm.models.abstract_vlm import AbstractVLMModel


class QwenVLModel(AbstractVLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_str = "qwen"
        self.batch_count = 0

        self.forbidden_name_patterns_for_wd.extend(
            # Add LoRA layers and the token embedding layer (which is also the LM head) to forbidden name patterns.
            ["lora", "embed_tokens", "lm_head"]
        )

    @property
    def module_attr_map(self) -> dict[str, str]:
        return {
            "vision_encoder": "model.vision_encoder",
            "adapter": "model.vision_adapter",
            "language_model": "model.language_model",
            "lm_head": "lm_head",
        }

    def setup_model(self):
        """Initialize model, processor and tokenizer"""

        if os.getenv("SLURM_JOB_ID") or os.getenv("LSB_JOBID"):
            cache_dir = os.getenv("HF_HUB_CACHE")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

        self.model = QwenVLForConditionalGeneration3D.from_pretrained(
            self.model_cfg.model_id,
            model_cfg=self.model_cfg.model_dump(),
            vision_embed_dim=self.vision_embed_dim,
            vision_embed_size=self.vision_embed_size,
            torch_dtype=torch.bfloat16
        )

        model: QwenVLModel3D = self.model.model
        if self.model_cfg.vision_encoder is not None:
            model.setup_vision_encoder()
            self.downsample_factor = self.model_cfg.vision_encoder.downsample_factor
        if self.model_cfg.adapter is not None:
            model.setup_vision_adapter()
            self.vision_adapter = model.vision_adapter

        self.processor = QwenVLProcessor3D.from_pretrained(
            self.model_cfg.model_id,
            cache_dir=cache_dir
        )
        self.tokenizer = self.processor.tokenizer

        self.model.post_init()

        return self.model

    def collate_fn(self, examples):
        """Custom collate function for Qwen model"""
        # Get the texts and images, and apply the chat template
        texts = [self.processor.apply_chat_template(example["formatted_qa"], tokenize=False) for example in examples]

        texts_prompt_only = [
            self.processor.apply_chat_template(example["formatted_qa_prompt_only"],
                                               tokenize=False) + "<|im_start|>assistant\n"
            for example in examples
        ]

        vision_inputs = None
        num_vision_tokens, vision_position_ids, adapter_args = None, None, {}

        if not self.text_only:
            vision_inputs = [
                example["vision_input"]
                for example in examples
            ]
            num_vision_tokens, vision_position_ids, adapter_args = self.vision_adapter.get_adapter_args(
                vision_inputs,
                self.vision_input_type,
                downsample_factor=self.downsample_factor
            )

            # This only works if we use homogenous image inputs!
            vision_inputs = torch.from_numpy(np.stack(vision_inputs))

        batch = self.processor(
            text=texts,
            num_vision_tokens=num_vision_tokens,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            do_truncation=True  # TODO: add truncation argument to configs
        )

        batch_prompt_only = self.processor(
            text=texts_prompt_only,
            num_vision_tokens=num_vision_tokens,
            return_tensors="pt",
            padding=True,
            padding_side="left"
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

    def get_model_specific_training_args(self):
        """Return model-specific training arguments"""
        return {
            "bf16": True,
            "tf32": True,
        }
