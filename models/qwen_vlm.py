import os
import torch
import transformers

from configs.model.model_config import ModelConfig
from qwenvl.processing_3d import QwenVLProcessor3D
from qwenvl.Qwen3D import QwenVLForConditionalGeneration3D
from utils import logger
from utils.filter_batch import extract_raw_samples_from_batch
from utils.vision_input_type import VisionInputType

from .abstract_vlm import AbstractVLMModel


class QwenVLModel(AbstractVLMModel):
    def __init__(
        self,
        model_cfg: ModelConfig,
        vision_input_type: str,
        local_files_only=False,
    ):
        super().__init__(model_cfg=model_cfg,
                         vision_input_type=vision_input_type,
                         local_files_only=local_files_only)
        self._model_str = "qwen"
        self.batch_count = 0

    @property
    def module_attr_map(self) -> dict[str, str]:
        return {
            "vision_encoder": "model.vision_encoder",
            "adapter": "model.vision_adapter",
            "language_model": "model.language_model",
            "lm_head": "lm_head",
        }

    def setup_model(self, bnb_config=None):
        """Initialize model, processor and PEFT configuration"""

        if os.getenv("SLURM_JOB_ID") or os.getenv("LSB_JOBID"):
            cache_dir = os.getenv("HF_HUB_CACHE")
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

        self.processor = QwenVLProcessor3D.from_pretrained(
            self.model_cfg.model_id,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
            reduce_block=self.model_cfg.adapter.reduce_block,
            vision_input_type=self.vision_input_type
        )
        self.tokenizer = self.processor.tokenizer

        self.model = QwenVLForConditionalGeneration3D.from_pretrained(
            self.model_cfg.model_id,
            model_cfg=self.model_cfg.model_dump(),
            torch_dtype=torch.bfloat16
        )
        
        if self.vision_input_type == VisionInputType.IMAGE:
            self.model.model.setup_vision_encoder()

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

        vision_inputs = [
            example["vision_input"]#.astype(np.float32) #TODO check why we had this cast before
            for example in examples
            if "vision_input" in example and example["vision_input"] is not None
        ]

        batch = self.processor(text=texts, vision_inputs=vision_inputs, return_tensors="pt", padding=True)

        batch_prompt_only = self.processor(
            text=texts_prompt_only,
            vision_inputs=vision_inputs,
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

        # Only keep the model's final assistant response as the prediction target.
        # Everything before and including the last "<|im_start|>assistant\n" is ignored (-100).
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
            "pixel_values_videos": batch["pixel_values_videos"],
            "video_grid_thw": batch["video_grid_thw"],
            "batch_count": self.batch_count,
            "raw_samples": raw_samples,
        }
        batch_out = transformers.BatchFeature(batch_out)
        self.batch_count += 1
        return batch_out

    # def get_trainable_params(self):
    #     """Return information about trainable parameters"""
    #     if self.train_only_vision_adapter:
    #         return {"peft_config": None, "train_only_vision_adapter": True}
    #     else:
    #         if self.train_vision_encoder:
    #             # Train both language model and vision encoder
    #             peft_config = LoraConfig(
    #                 task_type="CAUSAL_LM",
    #                 r=8,
    #                 lora_alpha=32,
    #                 target_modules=[
    #                     "q_proj", "k_proj", "v_proj", "o_proj",
    #                     "gate_proj", "up_proj", "down_proj",
    #                     "lm_head"
    #                 ],
    #                 modules_to_save=["vision_adapter"],
    #                 lora_dropout=0.1,
    #                 bias="none"
    #             )
    #         else:
    #             # Only train language model, exclude vision encoder modules
    #             peft_config = LoraConfig(
    #                 task_type="CAUSAL_LM",
    #                 r=8,
    #                 lora_alpha=32,
    #                 # Use specific module paths to target only language model components
    #                 # TODO use exclude_modules (only available in newest transformers dev right now)
    #                 target_modules=[
    #                     "model.language_model.layers.*.self_attn.q_proj",
    #                     "model.language_model.layers.*.self_attn.k_proj",
    #                     "model.language_model.layers.*.self_attn.v_proj",
    #                     "model.language_model.layers.*.self_attn.o_proj",
    #                     "model.language_model.layers.*.mlp.gate_proj",
    #                     "model.language_model.layers.*.mlp.up_proj",
    #                     "model.language_model.layers.*.mlp.down_proj",
    #                     "lm_head"
    #                 ],
    #                 modules_to_save=["vision_adapter"],
    #                 lora_dropout=0.1,
    #                 bias="none"
    #             )
    #
    #         return {"peft_config": peft_config, "train_only_vision_adapter": False}

    def get_model_specific_training_args(self):
        """Return model-specific training arguments"""
        return {
            "bf16": True,
            "tf32": True,
        }
