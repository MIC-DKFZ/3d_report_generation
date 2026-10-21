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
| `vlm/models/gemma3/modeling_gemma3_text_vl.py` | `models/gemma3/modeling_gemma3.py` |
| `vlm/models/phi4mini/modeling_phi4mini_vl.py` | `models/phi3/modeling_phi3.py` |
| `vlm/models/qwen3/modeling_qwen3_text_vl.py` | `models/qwen3/modeling_qwen3.py` |
| `vlm/models/qwen3vl/modeling_qwen3_vl.py` | `models/qwen3_vl/modeling_qwen3_vl.py` |
| `vlm/models/qwen3vl/processing_3d.py` | `models/qwen3_vl/processing_qwen3_vl.py` |
| `vlm/models/qwen25vl/modeling_qwen25_vl.py` | `models/qwen2_5_vl/modeling_qwen2_5_vl.py` |
| `vlm/models/qwen25vl/processing_3d.py` | `models/qwen2_5_vl/processing_qwen2_5_vl.py` |
| `vlm/training/trainers.py` | `trainer.py` |

The model and processor adaptations support 3D medical image conditioning.
The trainer adaptations support VLM logging, custom schedulers, checkpointing,
and supervised-token counting. Modification notices are also present in each file.
