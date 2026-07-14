from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoTokenizer

from vlm.models.abstract_vlm import AbstractVLMModel
from vlm.models.gemma3.Gemma3D import Gemma3TextVLForConditionalGeneration
from vlm.utils.filter_batch import extract_raw_samples_from_batch
from vlm.utils.helper import find_sublist, get_hf_hub_cache_dir


GEMMA3_VISION_PLACEHOLDER_TOKEN = "<image_soft_token>"


class Gemma3TextVLModel(AbstractVLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_str = "gemma3"
        self.batch_count = 0
        self._max_length = None

        self.forbidden_name_patterns_for_wd.extend(["lora", "embed_tokens", "lm_head"])

    @property
    def module_attr_map(self) -> dict[str, str]:
        return {
            "vision_encoder": "model.vision_encoder",
            "adapter": "model.vision_adapter",
            "language_model": "model.language_model",
            "lm_head": "lm_head",
        }

    def setup_model(self):
        """Initialize model, tokenizer and Gemma3 text VLM wrapper."""
        cache_dir = get_hf_hub_cache_dir()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.model_id,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = Gemma3TextVLForConditionalGeneration.from_pretrained(
            self.model_cfg.model_id,
            model_cfg=self.model_cfg.model_dump(),
            vision_embed_dim=self.vision_embed_dim,
            vision_embed_size=self.vision_embed_size,
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
            torch_dtype=torch.bfloat16,
        )
        self.model.config.image_token_id = self.tokenizer.convert_tokens_to_ids(
            GEMMA3_VISION_PLACEHOLDER_TOKEN
        )
        if self.model.config.image_token_id is None:
            raise ValueError(
                f"Gemma3 tokenizer does not contain the placeholder token "
                f"{GEMMA3_VISION_PLACEHOLDER_TOKEN!r}."
            )

        model = self.model.model
        if self.model_cfg.vision_encoder is not None:
            model.setup_vision_encoder()
            self.downsample_factor = model.vision_encoder.downsample_factor
        if self.model_cfg.adapter is not None:
            model.setup_vision_adapter()
            self.vision_adapter = model.vision_adapter

        self.processor = self.tokenizer
        return self.model

    @contextmanager
    def _padding_side(self, padding_side: str):
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        try:
            yield
        finally:
            self.tokenizer.padding_side = original_padding_side

    @staticmethod
    def _content_to_text(content) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, dict) and item.get("type") in {"image", "video"}:
                    parts.append(GEMMA3_VISION_PLACEHOLDER_TOKEN)
                else:
                    raise ValueError(f"Unsupported Gemma3 message content item: {item}")
            return "\n".join(part for part in parts if part)

        raise TypeError(f"Unsupported Gemma3 message content type: {type(content).__name__}")

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        return [
            {"role": message["role"], "content": self._content_to_text(message["content"])}
            for message in messages
        ]

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            self._normalize_messages(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _expand_vision_tokens(texts: list[str], num_vision_tokens: list[int] | None) -> list[str]:
        if num_vision_tokens is None:
            return texts

        if len(num_vision_tokens) != len(texts):
            raise ValueError(
                "Gemma3 text VLM currently supports exactly one vision input per sample. "
                f"Got {len(num_vision_tokens)} vision inputs for {len(texts)} text samples."
            )

        expanded = []
        for text, n_tokens in zip(texts, num_vision_tokens):
            n_placeholders = text.count(GEMMA3_VISION_PLACEHOLDER_TOKEN)
            if n_placeholders != 1:
                raise ValueError(
                    "Gemma3 text VLM currently expects exactly one image placeholder per sample. "
                    f"Found {n_placeholders} placeholders."
                )
            expanded.append(
                text.replace(
                    GEMMA3_VISION_PLACEHOLDER_TOKEN,
                    GEMMA3_VISION_PLACEHOLDER_TOKEN * int(n_tokens),
                    1,
                )
            )
        return expanded

    def _tokenize(self, texts: list[str], padding_side: str):
        if self.do_truncation:
            if self._max_length is None:
                if self.truncate_after is None or self.truncate_at is None:
                    raise ValueError(
                        "Gemma3 truncation requires both 'truncate_after' and 'truncate_at'"
                    )
                tokenized_marker = self.tokenizer(
                    self.truncate_after,
                    add_special_tokens=False,
                )["input_ids"]
                tokenized_sample = self.tokenizer(
                    texts[0],
                    add_special_tokens=False,
                )["input_ids"]
                marker_start_idx = find_sublist(tokenized_sample, tokenized_marker)
                assert marker_start_idx >= 0, "'truncate_after' not found in sample"
                self._max_length = marker_start_idx + len(tokenized_marker) + self.truncate_at
            truncation_kwargs = {"truncation": True, "max_length": self._max_length}
        else:
            truncation_kwargs = {"truncation": False}

        with self._padding_side(padding_side):
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                **truncation_kwargs,
            )

    def _build_vision_inputs(self, examples):
        raw_vision_inputs = [example["vision_input"] for example in examples]

        if any(isinstance(vision_input, (list, tuple)) for vision_input in raw_vision_inputs):
            raise ValueError(
                "Gemma3 text VLM currently supports report generation with one image per sample. "
                "Retrieval or multi-image prompts are not supported."
            )

        vision_inputs = [
            vi.detach().cpu().numpy() if isinstance(vi, torch.Tensor) else np.asarray(vi)
            for vi in raw_vision_inputs
        ]

        if len(vision_inputs) == 0:
            raise ValueError("Received an empty vision input list in Gemma3TextVLModel.collate_fn")

        (
            num_vision_tokens,
            vision_position_ids,
            adapter_args,
        ) = self.vision_adapter.get_adapter_args(
            vision_inputs,
            self.vision_input_type,
            downsample_factor=self.downsample_factor,
        )

        return (
            torch.from_numpy(np.stack(vision_inputs)),
            num_vision_tokens,
            vision_position_ids,
            adapter_args,
        )

    def collate_fn(self, examples):
        texts = [
            self._apply_chat_template(example["formatted_qa"], add_generation_prompt=False)
            for example in examples
        ]
        texts_prompt_only = [
            self._apply_chat_template(
                example["formatted_qa_prompt_only"],
                add_generation_prompt=True,
            )
            for example in examples
        ]

        vision_inputs = None
        num_vision_tokens, vision_position_ids, adapter_args = None, None, {}
        if not self.text_only:
            (
                vision_inputs,
                num_vision_tokens,
                vision_position_ids,
                adapter_args,
            ) = self._build_vision_inputs(examples)
            texts = self._expand_vision_tokens(texts, num_vision_tokens)
            texts_prompt_only = self._expand_vision_tokens(
                texts_prompt_only,
                num_vision_tokens,
            )

        batch = self._tokenize(texts, padding_side="right")
        batch_prompt_only = self._tokenize(texts_prompt_only, padding_side="left")

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        prompt_lengths = batch_prompt_only["attention_mask"].sum(dim=1)
        for i, prompt_length in enumerate(prompt_lengths.tolist()):
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
            "adapter_args": adapter_args,
        }
        self.batch_count += 1
        return batch_out

    def get_model_specific_training_args(self):
        return {
            "bf16": True,
            "tf32": True,
        }
