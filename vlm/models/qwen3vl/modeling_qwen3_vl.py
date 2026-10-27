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

from torch import nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration,
    Qwen3VLModel as HFQwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
)

from vlm.models.qwen_vl import AdapterVisionForConditionalGenerationMixin, AdapterVisionModelMixin


class Qwen3VLModel(AdapterVisionModelMixin, HFQwen3VLModel):
    base_model_prefix = ""
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    output_class = Qwen3VLModelOutputWithPast

    def __init__(self, config):
        # The HF model constructor also creates the unused native vision tower.
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.vision_adapter = None
        self.vision_encoder = None
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None     # cache rope_deltas here
        self.post_init()


class Qwen3VLForConditionalGeneration(
    AdapterVisionForConditionalGenerationMixin, HFQwen3VLForConditionalGeneration
):
    config: Qwen3VLConfig
    output_class = Qwen3VLCausalLMOutputWithPast

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
