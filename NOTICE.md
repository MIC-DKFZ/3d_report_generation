# Licensing and third-party notices

VLM project code, documentation, and configurations are licensed under Apache-2.0
in `LICENSE.md`. Adaptations retain applicable upstream copyright and
modification notices.

## Hugging Face Transformers adaptations (Apache-2.0)

Upstream project: https://github.com/huggingface/transformers
Copyright 2018- The Hugging Face team. All rights reserved.

Upstream paths below are relative to `src/transformers/`.

| VLM file | Upstream source |
| --- | --- |
| `vlm/models/qwen_vl.py` | `models/qwen2_5_vl/modeling_qwen2_5_vl.py`, `models/qwen3_vl/modeling_qwen3_vl.py` |
| `vlm/models/qwen3vl/modeling_qwen3_vl.py` | `models/qwen3_vl/modeling_qwen3_vl.py` |
| `vlm/models/qwen3vl/processing_3d.py` | `models/qwen3_vl/processing_qwen3_vl.py` |

The model and processor adaptations support 3D medical image conditioning.
Modification notices are also present in each file.
