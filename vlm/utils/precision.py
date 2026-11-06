"""
Precision of inference ('eval.inference_precision').

Greedy decoding in bfloat16 is not reproducible across batch sizes, GPU counts and GPU types:
rounding differences flip near-ties between tokens, after which the reports diverge (Yuan et al.,
"Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference",
arXiv:2506.09501). Float32 makes the reports much less sensitive to this, but not identical, also
for runs trained in bfloat16 autocast: upcasting bfloat16 weights to float32 is exact.

In either precision, the cuDNN backend of SDPA is disabled: its attention differs from run to run
for the same inputs (bfloat16 reports of the same volumes then differ between runs on Hopper GPUs),
while the other backends and FlashAttention are reproducible.
"""

import contextlib

import torch

from vlm.utils import logger

# Attention implementations that compute in float32; FlashAttention supports only 16-bit inputs.
FLOAT32_ATTENTION = ("sdpa", "eager")


def prepare_model_for_inference_precision(model: torch.nn.Module, precision: str) -> None:
    """
    Cast the model for 'precision' ('bf16' keeps it as loaded), and disable the cuDNN backend of
    SDPA, whose results are not reproducible.
    """
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"Unknown inference precision '{precision}'.")
    torch.backends.cuda.enable_cudnn_sdp(False)
    if precision == "bf16":
        return
    # TF32 matrix multiplications round the inputs to 10 mantissa bits. cuDNN convolutions keep
    # TF32, as when the training embeddings were made (also Qwen3.5's short convolutions).
    torch.backends.cuda.matmul.allow_tf32 = False
    model.float()
    attention = model.config._attn_implementation
    if attention not in FLOAT32_ATTENTION:
        logger.info_main(f"Inference in float32: attention '{attention}' replaced by 'sdpa'.")
        model.set_attn_implementation("sdpa")


def inference_precision_context(precision: str):
    """The autocast of 'precision' for generation: bfloat16 as in training, none in float32."""
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
