import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

from owl.model import OwnOwlModel
from owl.modeling_mplugowl3 import mPLUGOwl3Model
from utils import logger
from utils.filter_batch import extract_raw_samples_from_batch

from .abstract_vlm import AbstractVLMModel


class OwlVLModel(AbstractVLMModel):
    def __init__(
        self,
        model_id,
        train_only_vision_adapter: bool,
        output_dir="owl-vlm",
        vision_dim=864,
        embed_dim=896,
    ):
        super().__init__(model_id, train_only_vision_adapter)
        self.output_dir = output_dir
        self.vision_dim = vision_dim
        self.embed_dim = embed_dim
        self.batch_count = 0

    def setup_model(self, bnb_config=None):
        """Initialize model, processor, and tokenizer for Owl"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Load configuration
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)

        # Apply BNB config if provided
        if bnb_config:
            language_model = mPLUGOwl3Model.from_pretrained(
                self.model_id,
                config=config,
                attn_implementation="sdpa",
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            language_model = mPLUGOwl3Model.from_pretrained(
                self.model_id,
                config=config,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )

        # Initialize processor via the language model
        self.processor = language_model.init_processor(self.tokenizer)
        self.processor.inference_mode = False

        # Create our custom OwnOwlModel
        self.model = OwnOwlModel(
            config,
            language_model=language_model.language_model,
            vision_dim=self.vision_dim,
            embed_dim=self.embed_dim,
            tokenizer=self.tokenizer,
            debug=False,
        )

        # Only manually move to device if not using device_map
        if bnb_config is None:
            self.model.to(self.device)

        # Set trainable parameters
        self._set_trainable_params()

        return self.model

    def _set_trainable_params(self):
        """Set which parameters are trainable for the Owl model"""

        if self.train_only_vision_adapter:

            # Freeze all parameters first
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze only the vision2text_model parameters
            for param in self.model.vision2text_model.parameters():
                param.requires_grad = True

            # Optionally unfreeze HyperQwen2DecoderLayer parameters
            for m_module in self.model.modules():
                if "HyperQwen2DecoderLayer" in str(m_module.__class__):
                    if m_module.is_hyper_enabled:
                        for param in m_module.self_attn.parameters():
                            param.requires_grad = True

        # Print trainable parameters summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(
            f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%"
        )

    def pad_inputs(self, input_ids_list, input_ids_list_prompt_only, media_offset_list):
        """Helper function to pad inputs for Owl model"""
        # Pad text inputs using the tokenizer
        padded = self.tokenizer.pad(
            {"input_ids": input_ids_list}, return_tensors="pt", padding="longest"
        )
        padded_prompt_only = self.tokenizer.pad(
            {"input_ids": input_ids_list_prompt_only},
            return_tensors="pt",
            padding="longest",
        )
        pad_shape = padded["input_ids"].shape  # (batch_size, max_seq_len)

        padded_media_offsets = []
        for media_offset in media_offset_list:
            # Remove the batch dimension
            media_offset = media_offset.squeeze(0).to(torch.int32)
            # If the media offset is shorter than the pad width, pad it with the last row
            if media_offset.shape[0] < pad_shape[1]:
                pad_amount = pad_shape[1] - media_offset.shape[0]
                pad = torch.zeros(pad_amount, media_offset.shape[1], dtype=torch.int32)
                pad[:] = media_offset[-1, :]
                media_offset = torch.cat([media_offset, pad], dim=0)
            else:
                media_offset = media_offset[: pad_shape[1], :]
            padded_media_offsets.append(media_offset)
        media_offsets = torch.stack(padded_media_offsets, dim=0)
        return padded, padded_prompt_only, media_offsets

    def collate_fn(self, batch):
        """Custom collate function for Owl model"""
        image_tensors = []
        input_ids_list = []
        input_ids_list_prompt_only = []
        media_offset_list = []

        for sample in batch:

            messages = sample["formatted_qa"]
            messages_prompt_only = sample["formatted_qa_prompt_only"]

            # Flatten and transpose dimensions
            img_tensor = sample["embedding_tensor"]
            image_tensors.append(img_tensor)

            if "tokenized_qa_text_tokens" in sample:
                model_inputs = sample["tokenized_qa_text_tokens"]
                model_inputs_prompt_only = sample["tokenized_qa_text_prompt_only_tokens"]
            else:
                model_inputs = self.processor.encode_text_sft(
                    self.processor.build_text_qwen(messages)
                )
                model_inputs_prompt_only = self.processor.encode_text_sft(
                    self.processor.build_text_qwen(messages_prompt_only)
                )

            input_ids_list.append(model_inputs["input_ids"].squeeze(0))
            input_ids_list_prompt_only.append(model_inputs_prompt_only["input_ids"].squeeze(0))
            media_offset_list.append(model_inputs["media_offset"])

        images = torch.cat(image_tensors, dim=0)

        padded, padded_prompt_only, media_offsets = self.pad_inputs(
            input_ids_list, input_ids_list_prompt_only, media_offset_list
        )

        # Use input_ids as labels (mask pad tokens with -100)
        labels = padded["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # TODO: does this work for batch size > 1?
        media_offsets_prompt_only = media_offsets[:, : padded_prompt_only["input_ids"].shape[1]]

        raw_samples = extract_raw_samples_from_batch(batch)
        batch_out = {
            "image_embeds": images,
            "input_ids": padded["input_ids"],
            "input_ids_prompt_only": padded_prompt_only["input_ids"],
            "attention_mask": padded["attention_mask"],
            "attention_mask_prompt_only": padded_prompt_only["attention_mask"],
            "labels": labels,
            "media_offset": media_offsets,
            "media_offset_prompt_only": media_offsets_prompt_only,
            "batch_count": self.batch_count,
            "raw_samples": raw_samples,
        }
        batch_out = transformers.BatchFeature(batch_out)
        self.batch_count += 1

        return batch_out

    def get_trainable_params(self):
        """Return information about trainable parameters"""
        # No PEFT config for Owl - we use direct parameter freezing instead
        return {"peft_config": None, "train_only_vision_adapter": False}

    def get_model_specific_training_args(self):
        """Return model-specific training arguments"""
        return {
            "bf16": True,
        }
