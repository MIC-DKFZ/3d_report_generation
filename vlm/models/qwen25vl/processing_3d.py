from typing import Union

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLProcessorKwargs,
)
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class QwenVLProcessor3D(Qwen2_5_VLProcessor):
    """
    Only handles text processing and tokenization.
    """
    # attributes = ["image_processor", "tokenizer", "video_processor"]
    attributes = ["image_processor", "tokenizer"]

    image_processor_class = "AutoImageProcessor"
    # video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.max_length = None

    def _find_sublist(self, lis: list, sub: list):
        m = len(sub)
        for i in range(len(lis) - m + 1):
            if lis[i:i + m] == sub:
                return i
        return -1

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        num_vision_tokens: list[int] = None,
        do_truncation: bool = False,
        truncate_after: str | None = "<|im_start|>assistant",
        truncate_at: int | None = 939, # 3*median of entire train set, affects 66 cases
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, list):
            text = [text]
        text = text.copy()  # below lines change text in-place

        placeholder = "<|placeholder|>"
        if num_vision_tokens is not None:
            text_index = 0
            for i, n in enumerate(num_vision_tokens):
                if self.video_token in text[text_index]:
                    text[text_index] = text[text_index].replace(self.video_token, placeholder * n, 1)
                else:
                    text[text_index] = text[text_index].replace(placeholder, self.video_token)
                    text_index += 1
                    text[text_index] = text[text_index].replace(self.video_token, placeholder * n, 1)
            text[text_index] = text[text_index].replace(placeholder, self.video_token)




        if do_truncation and self.max_length is None:
            # TODO: we assume that 'truncate_after' appears at the same position across the entire dataset
            _tokenized_truncate_after = self.tokenizer(truncate_after)["input_ids"]

            return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
            return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
            _text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

            _tokenized_sample = _text_inputs["input_ids"][0]
            marker_start_idx = self._find_sublist(_tokenized_sample, _tokenized_truncate_after)
            assert marker_start_idx >= 0, f"'truncate_after' not found in sample"
            self.max_length = marker_start_idx + len(_tokenized_truncate_after) + truncate_at
        else:
            return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
            return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)

        # Original Qwen25VL stuff
        # text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], truncation=do_truncation, max_length=self.max_length)
        if not do_truncation:
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], truncation=False)
        else:
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], truncation=True, max_length=self.max_length)

        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs}, tensor_type=return_tensors)

