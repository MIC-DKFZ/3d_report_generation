import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from qwenvl.processing_3d import Qwen2VLProcessor3D
from qwenvl.Qwen3D import Qwen2VLForConditionalGeneration3D

from .abstract_vlm import AbstractVLMModel


class QwenVLModel(AbstractVLMModel):
    def __init__(
        self, model_id, output_dir="qwen2-vlm", train_only_vision_adapter=False
    ):
        super().__init__(model_id, train_only_vision_adapter)
        self.output_dir = output_dir
        self.batch_count = 0

    def setup_model(self, bnb_config=None):
        """Initialize model, processor and PEFT configuration"""
        # Initialize processor
        self.processor = Qwen2VLProcessor3D.from_pretrained(
            self.model_id,
            cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
            local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer

        if self.train_only_vision_adapter:
            # For adapter-only training, don't use quantization for the adapter
            self.model = Qwen2VLForConditionalGeneration3D.from_pretrained(
                self.model_id, device_map="auto", torch_dtype=torch.bfloat16
            )

            # Freeze all parameters except vision adapter
            self._set_trainable_params()
            return self.model
        else:
            # Use quantization if provided
            if bnb_config:
                self.model = Qwen2VLForConditionalGeneration3D.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    quantization_config=bnb_config,
                )
            else:
                self.model = Qwen2VLForConditionalGeneration3D.from_pretrained(
                    self.model_id, device_map="auto", torch_dtype=torch.bfloat16
                )

            return self.model

    def _set_trainable_params(self):
        """Set which parameters are trainable based on configuration"""
        if self.train_only_vision_adapter:
            # Freeze all parameters
            for name, param in self.model.named_parameters():
                if (
                    "vision_adapter" not in name
                ):  # Freeze everything except vision adapter
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    print(f"Unfreezing: {name}")

            # Print trainable parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(
                f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%"
            )

    def collate_fn(self, examples):
        """Custom collate function for Qwen model"""
        # Get the texts and images, and apply the chat template
        texts = [
            self.processor.apply_chat_template(example["formatted_qa"], tokenize=False)
            for example in examples
        ]

        embeddings = [example["embedding"] for example in examples]

        batch = self.processor(
            text=texts, vision_embeddings=embeddings, return_tensors="pt", padding=True
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        if isinstance(self.processor, Qwen2VLProcessor3D):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [
                self.processor.tokenizer.convert_tokens_to_ids(
                    self.processor.image_token
                )
            ]

        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        video_tokens = [
            self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)
        ]
        for video_token_id in video_tokens:
            labels[labels == video_token_id] = -100

        batch["labels"] = labels
        self.batch_count += 1
        return batch

    def get_trainable_params(self):
        """Return information about trainable parameters"""
        if self.train_only_vision_adapter:
            return {"peft_config": None, "train_only_vision_adapter": True}
        else:
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.05,
                r=8,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            )
            return {"peft_config": peft_config, "train_only_vision_adapter": False}

    def get_model_specific_training_args(self):
        """Return model-specific training arguments"""
        return {
            "bf16": True,
            "tf32": True,
        }
