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
from transformers.conversion_mapping import WeightRenaming, register_checkpoint_conversion_mapping
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration as HFQwen25VLForConditionalGeneration,
    Qwen2_5_VLModel as HFQwen25VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLTextModel,
)

from vlm.models.qwen_vl import AdapterVisionForConditionalGenerationMixin, AdapterVisionModelMixin


class Qwen25VLModel(AdapterVisionModelMixin, HFQwen25VLModel):
    base_model_prefix = ""
    config: Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    output_class = Qwen2_5_VLModelOutputWithPast

    def __init__(self, config):
        # The HF model constructor also creates the unused native vision tower.
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.vision_adapter = None
        self.vision_encoder = None
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here
        self.post_init()


class Qwen25VLForConditionalGeneration(
    AdapterVisionForConditionalGenerationMixin, HFQwen25VLForConditionalGeneration
):
    config: Qwen2_5_VLConfig
    output_class = Qwen2_5_VLCausalLMOutputWithPast

    def __init__(self, config):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.model = Qwen25VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


# Hub checkpoints store the decoder under 'model.*' and the native vision tower under 'visual.*'.
# Transformers skips its own Qwen2.5-VL renaming for custom classes, so register it for ours.
register_checkpoint_conversion_mapping(
    "Qwen25VLForConditionalGeneration",
    [
        WeightRenaming(source_patterns=r"^visual", target_patterns="model.visual"),
        WeightRenaming(
            source_patterns=r"^model(?!\.(language_model|visual|vision_adapter|vision_encoder))",
            target_patterns="model.language_model",
        ),
    ],
    overwrite=True,
)
