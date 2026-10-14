# Licensing and third-party notices

VLM contains material under two licenses. Original project code, documentation,
and configurations are covered by CC BY-SA 4.0 in `LICENSE.md`, except for the
Apache-2.0 files listed below. Those files, including their VLM modifications,
are covered by `licenses/Apache-2.0.txt` and retain applicable upstream notices.
The package's `CC-BY-SA-4.0 AND Apache-2.0` expression describes the combined
distribution; it does not apply both licenses to every file or offer a choice
between them.

## Hugging Face Transformers adaptations (Apache-2.0)

Upstream project: https://github.com/huggingface/transformers
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
