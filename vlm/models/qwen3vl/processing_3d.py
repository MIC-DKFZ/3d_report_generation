# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified to support VLM's 3D medical image conditioning and custom training workflow.

from typing import Union
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class QwenVLProcessor3D(Qwen3VLProcessor):
    """
    Only handles text processing and tokenization.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        num_vision_tokens: list[int] = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
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

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        # Token type IDs mark native image/video tokens, which this processor does not use.
        output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], truncation=False)

        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        return BatchFeature(data={**text_inputs}, tensor_type=return_tensors)
