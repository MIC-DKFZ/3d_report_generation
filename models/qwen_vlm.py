import os
from pathlib import Path

import torch
import transformers
from peft import LoraConfig

from qwenvl.processing_3d import QwenVLProcessor3D
from qwenvl.Qwen3D import QwenVLForConditionalGeneration3D
from utils.filter_batch import extract_raw_samples_from_batch

from .abstract_vlm import AbstractVLMModel
from utils import logger

import numpy as np


class QwenVLModel(AbstractVLMModel):
    def __init__(
        self,
        model_id,
        output_dir="qwen25-vlm",
        train_only_vision_adapter=False,
        train_vision_encoder=True,
        vision_encoder_path=None,
        reduce_block=[2,2,2],
        # base_model_id=None,
        checkpoint_path=None,
    ):
        super().__init__(model_id, train_only_vision_adapter, train_vision_encoder, vision_encoder_path)
        self._model_str = "qwen"
        self.output_dir = output_dir
        self.batch_count = 0
        self.reduce_block = reduce_block
        # if base_model_id is None:
        #     self.base_model_id = model_id
        # else:
        #     self.base_model_id = base_model_id

        self.checkpoint_path = checkpoint_path

    def setup_model(self, bnb_config=None):
        """Initialize model, processor and PEFT configuration"""
        
        self.processor = QwenVLProcessor3D.from_pretrained(
            self.model_id,
            cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
            local_files_only=True,
            reduce_block=self.reduce_block,
            train_vision_encoder = self.train_vision_encoder
        )
        self.tokenizer = self.processor.tokenizer

        if self.checkpoint_path:
            # If a checkpoint path is provided, load the model from the checkpoint
            logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
            self.model = QwenVLForConditionalGeneration3D.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                local_files_only=True,
                reduce_block=self.reduce_block,
                train_vision_encoder=self.train_vision_encoder,
                vision_encoder_path=self.vision_encoder_path
            )

        elif bnb_config and not self.train_only_vision_adapter:
            self.model = QwenVLForConditionalGeneration3D.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                reduce_block=self.reduce_block,
                train_vision_encoder=self.train_vision_encoder,
                vision_encoder_path=self.vision_encoder_path
            )
        else:
            self.model = QwenVLForConditionalGeneration3D.from_pretrained(
                self.model_id, 
                torch_dtype=torch.bfloat16,
                reduce_block=self.reduce_block,
                train_vision_encoder=self.train_vision_encoder,
                vision_encoder_path=self.vision_encoder_path
            )

        if self.train_vision_encoder:
            self.model.model.setup_vision_encoder(self.vision_encoder_path)

        if self.train_only_vision_adapter:
            # For adapter-only training, don't use quantization for the adapter
            # Freeze all parameters except vision adapter
            self._set_trainable_params()

        if self.checkpoint_path:
            self.model.load_adapter(self.checkpoint_path)

        return self.model

    def _set_trainable_params(self):
        """Set which parameters are trainable based on configuration"""
        if self.train_only_vision_adapter:
            # Freeze all parameters
            for name, param in self.model.named_parameters():
                if "vision_adapter" not in name:  # Freeze everything except vision adapter
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    logger.debug(f"Unfreezing: {name}")

            # Print trainable parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

    def collate_fn(self, examples):
        """Custom collate function for Qwen model"""
        # Get the texts and images, and apply the chat template
        texts = [self.processor.apply_chat_template(example["formatted_qa"], tokenize=False) for example in examples]

        texts_prompt_only = [
            self.processor.apply_chat_template(example["formatted_qa_prompt_only"], tokenize=False)
            for example in examples
        ]

        embeddings = [
            example["embedding"]#.astype(np.float32) #TODO check why we had this cast before
            for example in examples
            if "embedding" in example and example["embedding"] is not None
        ]

        batch = self.processor(text=texts, vision_embeddings=embeddings, return_tensors="pt", padding=True)

        batch_prompt_only = self.processor(
            text=texts_prompt_only,
            vision_embeddings=embeddings,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        if isinstance(self.processor, QwenVLProcessor3D):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]

        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        video_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)]
        for i in range(labels.shape[0]):
            video_positions = (labels[i] == video_tokens[0]).nonzero(as_tuple=True)[0]
            if len(video_positions) > 0:
                first_video_pos = video_positions[0]
                labels[i, :first_video_pos + 1] = -100
        for video_token_id in video_tokens:
            labels[labels == video_token_id] = -100

        raw_samples = extract_raw_samples_from_batch(examples)
        batch_out = {
            "input_ids": batch["input_ids"],
            "input_ids_prompt_only": batch_prompt_only["input_ids"],
            "attention_mask": batch["attention_mask"],
            "attention_mask_prompt_only": batch_prompt_only["attention_mask"],
            "labels": labels,
            "pixel_values_videos": batch["pixel_values_videos"],
            "video_grid_thw": batch["video_grid_thw"],
            "batch_count": self.batch_count,
            "raw_samples": raw_samples,
        }
        batch_out = transformers.BatchFeature(batch_out)
        self.batch_count += 1
        return batch_out

    def get_trainable_params(self):
        """Return information about trainable parameters"""
        if self.train_only_vision_adapter:
            return {"peft_config": None, "train_only_vision_adapter": True}
        else:
            if self.train_vision_encoder:
                peft_config = LoraConfig(
                    task_type="CAUSAL_LM",
                    r=8,
                    lora_alpha=32,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "lm_head"
                    ],
                    #exclude_modules=["vision_adapter"],
                    modules_to_save=["vision_adapter"],
                    lora_dropout=0.1,
                    bias="none"
                )
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
