"""Exercise the shared Transformers runtime without downloading model weights."""


import pytest
import torch
from transformers import (
    Gemma3TextConfig,
    Phi3Config,
    Qwen2_5_VLConfig,
    Qwen3_5Config,
    Qwen3Config,
    Qwen3VLConfig,
)

from vlm.models.gemma3.modeling_gemma3_text_vl import Gemma3TextVLForConditionalGeneration
from vlm.models.phi4mini.modeling_phi4mini_vl import Phi4MiniVLForConditionalGeneration
from vlm.models.qwen25vl.modeling_qwen25_vl import Qwen25VLForConditionalGeneration
from vlm.models.qwen3.modeling_qwen3_text_vl import Qwen3TextVLForConditionalGeneration
from vlm.models.qwen35.modeling_qwen35 import Qwen35VLForConditionalGeneration
from vlm.models.qwen3vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


BACKBONES = ["qwen25", "qwen3vl", "qwen35", "qwen3", "phi4", "gemma3"]
VISION_TOKENS = [4, 6, 7]


def make_config(backbone):
    text = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    special = dict(
        video_token_id=4, image_token_id=4, vision_start_token_id=6, vision_end_token_id=7
    )
    if backbone in ("qwen25", "qwen3vl", "qwen35"):
        text["rope_parameters"] = {"rope_type": "default", "mrope_section": [1, 1, 2]}
        if backbone == "qwen35":
            # One recurrent linear-attention layer before one full-attention layer.
            text.update(
                num_hidden_layers=2,
                layer_types=["linear_attention", "full_attention"],
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                linear_key_head_dim=4,
                linear_value_head_dim=4,
                linear_conv_kernel_dim=2,
            )
            text["rope_parameters"].update(mrope_interleaved=True, partial_rotary_factor=1.0)
        config_class = {"qwen25": Qwen2_5_VLConfig, "qwen3vl": Qwen3VLConfig, "qwen35": Qwen3_5Config}
        config = config_class[backbone](text_config=text, **special)
    else:
        config_class = {"qwen3": Qwen3Config, "phi4": Phi3Config, "gemma3": Gemma3TextConfig}[
            backbone
        ]
        if backbone == "phi4":
            text["original_max_position_embeddings"] = 128
        config = config_class(**text, **special)
    config._attn_implementation = "eager"
    return config


def make_model(backbone):
    model_class = {
        "qwen25": Qwen25VLForConditionalGeneration,
        "qwen3vl": Qwen3VLForConditionalGeneration,
        "qwen35": Qwen35VLForConditionalGeneration,
        "qwen3": Qwen3TextVLForConditionalGeneration,
        "phi4": Phi4MiniVLForConditionalGeneration,
        "gemma3": Gemma3TextVLForConditionalGeneration,
    }[backbone]
    torch.manual_seed(13)
    return model_class(make_config(backbone)).eval()


def vision_batch():
    # Two left-padded samples with different numbers of image tokens.
    ids = torch.tensor([[0, 8, 6, 4, 4, 7, 9], [8, 6, 4, 4, 4, 7, 9]])
    return dict(
        input_ids=ids,
        attention_mask=ids.ne(0).long(),
        pixel_values_videos=torch.randn(5, 16),
        vision_position_ids=[
            torch.tensor([[0, 0], [0, 0], [0, 1]]),
            torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        ],
    )


@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbone_training_and_cached_generation(backbone):
    model = make_model(backbone)
    batch = vision_batch()
    batch["pixel_values_videos"].requires_grad_()
    labels = batch["input_ids"].clone()
    labels[:, :-1] = -100
    output = model(**batch, labels=labels, num_items_in_batch=torch.tensor(2), use_cache=False)
    # With labels, logits are computed only at the target positions: (num_targets, vocab).
    targets = labels[:, 1:][labels[:, 1:] != -100]
    assert output.logits.shape == (targets.numel(), 32)
    expected = torch.nn.functional.cross_entropy(output.logits, targets, reduction="sum") / 2
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    assert torch.isfinite(batch["pixel_values_videos"].grad).all()
    assert batch["pixel_values_videos"].grad.abs().sum() > 0

    # Cached decoding must agree with recomputing the full multimodal prefix.
    options = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
    cached = model.generate(**batch, use_cache=True, **options)
    uncached = model.generate(**batch, use_cache=False, **options)
    assert cached.shape == (2, 10)
    torch.testing.assert_close(cached, uncached)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_generation_steps_match_uncached_forward(backbone):
    model = make_model(backbone)
    batch = vision_batch()
    generated = model.generate(
        **batch,
        max_new_tokens=3,
        min_new_tokens=3,
        do_sample=False,
        pad_token_id=0,
        suppress_tokens=VISION_TOKENS,
        output_logits=True,
        return_dict_in_generate=True,
    )
    prompt_length = batch["input_ids"].shape[1]
    for step, logits in enumerate(generated.logits):
        # Each decoding step must see the vision input and positions of a fresh forward pass.
        attention_mask = torch.cat(
            [batch["attention_mask"], torch.ones(2, step, dtype=torch.long)], dim=1
        )
        inputs = dict(
            batch,
            input_ids=generated.sequences[:, : prompt_length + step],
            attention_mask=attention_mask,
        )
        if backbone not in ("qwen25", "qwen3vl", "qwen35"):
            # Text backbones use 1D positions that skip left padding.
            inputs["position_ids"] = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        expected = model(**inputs, use_cache=False).logits[:, -1]
        torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-5)
