"""Exercise the shared Transformers runtime without downloading model weights."""

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoConfig,
    Gemma3TextConfig,
    Phi3Config,
    Qwen2_5_VLConfig,
    Qwen2TokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3Config,
    Qwen3VLConfig,
)

from vlm.models.gemma3.modeling_gemma3_text_vl import Gemma3TextVLForConditionalGeneration
from vlm.models.phi4mini.modeling_phi4mini_vl import Phi4MiniVLForConditionalGeneration
from vlm.models.qwen25vl.modeling_qwen25_vl import Qwen25VLForConditionalGeneration
from vlm.models.qwen25vl.processing_3d import QwenVLProcessor3D as Qwen25Processor
from vlm.models.qwen3.modeling_qwen3_text_vl import Qwen3TextVLForConditionalGeneration
from vlm.models.qwen3vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from vlm.models.qwen3vl.processing_3d import QwenVLProcessor3D as Qwen3Processor
from vlm.training.checkpoint_state import load_model_checkpoint


def make_model(backbone):
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
    if backbone in ("qwen25", "qwen3vl"):
        text["rope_scaling"] = {"rope_type": "default", "mrope_section": [1, 1, 2]}
        config_class, model_class = (
            (Qwen2_5_VLConfig, Qwen25VLForConditionalGeneration)
            if backbone == "qwen25"
            else (Qwen3VLConfig, Qwen3VLForConditionalGeneration)
        )
        config = config_class(text_config=text, **special)
    else:
        config_class, model_class = {
            "qwen3": (Qwen3Config, Qwen3TextVLForConditionalGeneration),
            "phi4": (Phi3Config, Phi4MiniVLForConditionalGeneration),
            "gemma3": (Gemma3TextConfig, Gemma3TextVLForConditionalGeneration),
        }[backbone]
        if backbone == "phi4":
            text["original_max_position_embeddings"] = 128
        config = config_class(**text, **special)
    config._attn_implementation = "eager"
    torch.manual_seed(13)
    return model_class(config).eval()


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


@pytest.mark.parametrize("backbone", ["qwen25", "qwen3vl", "qwen3", "phi4", "gemma3"])
def test_backbone_training_and_cached_generation(backbone):
    model = make_model(backbone)
    batch = vision_batch()
    batch["pixel_values_videos"].requires_grad_()
    labels = batch["input_ids"].clone()
    labels[:, :-1] = -100
    output = model(**batch, labels=labels, num_items_in_batch=torch.tensor(2), use_cache=False)
    expected = (
        torch.nn.functional.cross_entropy(
            output.logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1), reduction="sum"
        )
        / 2
    )
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


def test_qwen25_save_load_and_legacy_checkpoint(tmp_path):
    from safetensors.torch import save_file

    model = make_model("qwen25")
    model.save_pretrained(tmp_path / "current")
    config = AutoConfig.from_pretrained(tmp_path / "current", local_files_only=True)
    assert config.model_type == "qwen2_5_vl"
    reloaded = Qwen25VLForConditionalGeneration.from_pretrained(
        tmp_path / "current", local_files_only=True, attn_implementation="eager"
    ).eval()
    batch = vision_batch()
    torch.testing.assert_close(model(**batch).logits, reloaded(**batch).logits)

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    weights = {
        key.replace("model.language_model.", "model."): value.contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(weights, legacy / "model.safetensors")
    # Original Qwen2.5 releases stored text settings at the config root and
    # decoder weights directly under model.*, rather than model.language_model.*.
    legacy_config = model.config.to_dict()
    legacy_config.update(legacy_config.pop("text_config"))
    legacy_config["model_type"] = "qwen2_5_vl"
    (legacy / "config.json").write_text(json.dumps(legacy_config))
    converted = Qwen25VLForConditionalGeneration.from_pretrained(
        legacy, local_files_only=True, attn_implementation="eager"
    ).eval()
    torch.testing.assert_close(model(**batch).logits, converted(**batch).logits)

    with torch.no_grad():
        reloaded.model.language_model.embed_tokens.weight.zero_()
    _, unexpected = load_model_checkpoint(reloaded, legacy)
    assert not unexpected
    torch.testing.assert_close(model(**batch).logits, reloaded(**batch).logits)


@pytest.mark.parametrize("processor_class", [Qwen25Processor, Qwen3Processor])
def test_processor_roundtrip_expands_custom_vision_tokens(tmp_path, processor_class):
    vocab = {"[UNK]": 0, "[PAD]": 1, "<|image_pad|>": 2, "<|video_pad|>": 3}
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<|image_pad|>", "<|video_pad|>"],
    )
    processor = processor_class(image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer)
    processor.save_pretrained(tmp_path)
    reloaded = processor_class.from_pretrained(tmp_path, local_files_only=True)
    inputs = reloaded(
        text=["<|video_pad|>", "<|video_pad|><|video_pad|>"],
        num_vision_tokens=[2, 1, 3],
        padding=True,
        return_tensors="pt",
    )
    assert (inputs["input_ids"] == 3).sum(dim=1).tolist() == [2, 4]


def test_shared_registry_registers_nnfoundation_backbone(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VLM_OUTPUT_DIR", str(tmp_path))
    from vlm.models.qwen25vl.wrapper import Qwen25VLWrapper
    from vlm.registries.vlm_registry import VLM_REGISTRY

    assert VLM_REGISTRY["Qwen2.5-VL-3B"].model_class is Qwen25VLWrapper
    assert {"Qwen3-1.7B", "Qwen3-VL-2B", "Phi4Mini", "Gemma3-4B"} <= VLM_REGISTRY.keys()
