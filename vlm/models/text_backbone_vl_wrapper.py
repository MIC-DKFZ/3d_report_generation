from contextlib import contextmanager
from typing import ClassVar

import torch
from transformers import AutoTokenizer, PreTrainedModel

from vlm.models.abstract_vl_wrapper import AbstractVLWrapper
from vlm.data.batches import extract_raw_samples_from_batch
from vlm.utils.paths import get_hf_hub_cache_dir


class TextBackboneVLWrapper(AbstractVLWrapper):
    """Framework interface for text backbones augmented with a vision encoder and adapter.

    Subclasses supply their model class, name, and placeholder token. They can customize
    tokenizer setup and chat formatting while sharing the single-image batch pipeline.
    """

    model_class: ClassVar[type[PreTrainedModel]]
    model_name: ClassVar[str]
    vision_placeholder_token: ClassVar[str]
    vision_token_id_attr: ClassVar[str] = "video_token_id"
    extra_forbidden_name_patterns_for_wd: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_str = self.model_name
        self.batch_count = 0
        self.forbidden_name_patterns_for_wd.extend(self.extra_forbidden_name_patterns_for_wd)

    def setup_model(self):
        """Load the tokenizer and model, then initialize the configured vision components."""
        cache_dir = get_hf_hub_cache_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.model_id.get_resolved_path(),
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
        )
        self._configure_tokenizer()

        self.model = self.model_class.from_pretrained(
            self.model_cfg.model_id.get_resolved_path(),
            cache_dir=cache_dir,
            local_files_only=self.local_files_only,
            dtype=torch.bfloat16,
        )
        token_id = self.tokenizer.convert_tokens_to_ids(self.vision_placeholder_token)
        if token_id is None:
            raise ValueError(
                f"{self.model_name} tokenizer does not contain the placeholder token "
                f"{self.vision_placeholder_token!r}."
            )
        setattr(self.model.config, self.vision_token_id_attr, token_id)

        self._setup_vision_components()
        if self.model_cfg.vision_encoder is not None:
            self.downsample_factor = self.model.model.vision_encoder.downsample_factor

        self.processor = self.tokenizer
        return self.model

    def _configure_tokenizer(self) -> None:
        """Apply any model-specific tokenizer adjustments before loading the model."""

    @contextmanager
    def _padding_side(self, padding_side: str):
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        try:
            yield
        finally:
            self.tokenizer.padding_side = original_padding_side

    def _content_to_text(self, content) -> str:
        """Render neutral message content using the model-specific vision placeholder."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, dict) and item.get("type") == "vision":
                    parts.append(self.vision_placeholder_token)
                else:
                    raise ValueError(f"Unsupported {self.model_name} message content item: {item}")
            return "\n".join(part for part in parts if part)

        raise TypeError(
            f"Unsupported {self.model_name} message content type: {type(content).__name__}"
        )

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        return [
            {"role": message["role"], "content": self._content_to_text(message["content"])}
            for message in messages
        ]

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return super()._apply_chat_template(
            self._normalize_messages(messages),
            add_generation_prompt=add_generation_prompt,
        )

    def _expand_vision_tokens(
        self, texts: list[str], num_vision_tokens: list[int] | None
    ) -> list[str]:
        if num_vision_tokens is None:
            return texts

        if len(num_vision_tokens) != len(texts):
            raise ValueError(
                f"{self.model_name} currently supports exactly one vision input per sample. "
                f"Got {len(num_vision_tokens)} vision inputs for {len(texts)} text samples."
            )

        expanded = []
        for text, n_tokens in zip(texts, num_vision_tokens):
            n_placeholders = text.count(self.vision_placeholder_token)
            if n_placeholders != 1:
                raise ValueError(
                    f"{self.model_name} currently expects exactly one vision placeholder "
                    "per sample. "
                    f"Found {n_placeholders} placeholders."
                )
            expanded.append(
                text.replace(
                    self.vision_placeholder_token,
                    self.vision_placeholder_token * int(n_tokens),
                    1,
                )
            )
        return expanded

    def _tokenize(self, texts: list[str], padding_side: str):
        with self._padding_side(padding_side=padding_side):
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

    def _build_vision_inputs(self, examples):
        raw_vision_inputs = [example["vision_input"] for example in examples]
        if any(isinstance(vision_input, (list, tuple)) for vision_input in raw_vision_inputs):
            raise ValueError(
                f"{self.model_name} currently supports report generation with one image "
                "per sample. "
                "Retrieval or multi-image prompts are not supported."
            )

        return self._prepare_vision_inputs(raw_vision_inputs)

    def collate_fn(self, examples):
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
            (
                vision_inputs,
                num_vision_tokens,
                vision_position_ids,
                adapter_args,
            ) = self._build_vision_inputs(examples)
            texts = self._expand_vision_tokens(texts, num_vision_tokens)
            texts_prompt_only = self._expand_vision_tokens(texts_prompt_only, num_vision_tokens)

        batch_prompt_only = self._tokenize(texts_prompt_only, padding_side="left")
        prompt_lengths = batch_prompt_only["attention_mask"].sum(dim=1).tolist()
        batch = self.tokenizer(texts, padding=False, truncation=False)
        batch = self._truncate_and_pad(batch, prompt_lengths)

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, prompt_length in enumerate(prompt_lengths):
            labels[i, :prompt_length] = -100

        batch_out = {
            "input_ids": batch["input_ids"],
            "input_ids_prompt_only": batch_prompt_only["input_ids"],
            "attention_mask": batch["attention_mask"],
            "attention_mask_prompt_only": batch_prompt_only["attention_mask"],
            "labels": labels,
            "pixel_values_videos": vision_inputs,
            "batch_count": self.batch_count,
            "raw_samples": extract_raw_samples_from_batch(examples),
            "vision_position_ids": vision_position_ids,
            "adapter_args": adapter_args,
        }
        self.batch_count += 1
        return batch_out
