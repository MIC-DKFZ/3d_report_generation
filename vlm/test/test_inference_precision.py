import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from vlm.configs.evaluation_config import EvaluationConfig
from vlm.utils.precision import inference_precision_context, prepare_model_for_inference_precision


def tiny_model(attention="eager"):
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2,
    )
    config._attn_implementation = attention
    return LlamaForCausalLM(config).to(torch.bfloat16)


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_the_cudnn_attention_backend_is_disabled(monkeypatch, precision):
    # Its results differ from run to run for the same inputs.
    calls = []
    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", calls.append)
    prepare_model_for_inference_precision(tiny_model("sdpa"), precision)
    assert calls == [False]


def test_bf16_keeps_the_model_and_autocasts():
    model = tiny_model()
    prepare_model_for_inference_precision(model, "bf16")
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert isinstance(inference_precision_context("bf16"), torch.autocast)


def test_fp32_upcasts_exactly_and_disables_tf32(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    model = tiny_model("sdpa")
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    prepare_model_for_inference_precision(model, "fp32")
    for name, p in model.named_parameters():
        assert p.dtype == torch.float32 and torch.equal(p, weights[name].float())
    assert model.config._attn_implementation == "sdpa"
    assert not torch.backends.cuda.matmul.allow_tf32
    # Convolutions (the embedding encoder) keep their TF32 setting, as for the training embeddings.
    assert torch.backends.cudnn.allow_tf32
    with inference_precision_context("fp32"):
        assert not torch.is_autocast_enabled("cuda")


def test_unknown_precisions_are_rejected():
    with pytest.raises(ValueError):
        prepare_model_for_inference_precision(tiny_model(), "fp16")
    with pytest.raises(ValueError):
        EvaluationConfig(inference_precision="fp16")
    assert EvaluationConfig().inference_precision == "bf16"
