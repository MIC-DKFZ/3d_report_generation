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
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration as HFQwen35ForConditionalGeneration,
    Qwen3_5Model as HFQwen35Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
)

from vlm.models.qwen_vl import AdapterVisionForConditionalGenerationMixin, AdapterVisionModelMixin


class Qwen35VLModel(AdapterVisionModelMixin, HFQwen35Model):
    base_model_prefix = ""
    config: Qwen3_5Config
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"]
    output_class = Qwen3_5ModelOutputWithPast

    def __init__(self, config):
        # The HF model constructor also creates the unused native vision tower.
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.vision_adapter = None
        self.vision_encoder = None
        self.language_model = Qwen3_5TextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here
        self.post_init()


class Qwen35VLForConditionalGeneration(
    AdapterVisionForConditionalGenerationMixin, HFQwen35ForConditionalGeneration
):
    config: Qwen3_5Config
    output_class = Qwen3_5CausalLMOutputWithPast
    # Hub checkpoints also store the native vision tower and the multi-token prediction head.
    _keys_to_ignore_on_load_unexpected = [r"^model\.visual\.", r"^mtp\."]

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Qwen35VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
