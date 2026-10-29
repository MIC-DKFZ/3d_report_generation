"""Exercise the shared Transformers runtime without downloading model weights."""

import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoConfig,
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2_5_VLConfig,
    Qwen3_5Config,
    Qwen2Tokenizer,
    Qwen2VLImageProcessor,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3VLConfig,
    SiglipVisionConfig,
)

from vlm.models.gemma3.modeling_gemma3_text_vl import Gemma3TextVLForConditionalGeneration
from vlm.models.phi4mini.modeling_phi4mini_vl import Phi4MiniVLForConditionalGeneration
from vlm.models.qwen25vl.modeling_qwen25_vl import Qwen25VLForConditionalGeneration
from vlm.models.qwen25vl.processing_3d import QwenVLProcessor3D as Qwen25Processor
from vlm.models.qwen3.modeling_qwen3_text_vl import Qwen3TextVLForConditionalGeneration
from vlm.models.qwen35.modeling_qwen35 import Qwen35VLForConditionalGeneration
from vlm.models.qwen3vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from vlm.models.qwen3vl.processing_3d import QwenVLProcessor3D as Qwen3Processor


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


@pytest.mark.parametrize("backbone", ["qwen3", "phi4", "gemma3"])
def test_text_backbone_loading_keeps_pretrained_tied_weights(tmp_path, backbone):
    text_config = make_config(backbone)
    text_config.tie_word_embeddings = True
    torch.manual_seed(5)
    if backbone == "gemma3":
        config = Gemma3Config(
            text_config=text_config,
            vision_config=SiglipVisionConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=8,
                patch_size=4,
            ),
            mm_tokens_per_image=4,
        )
        pretrained = Gemma3ForConditionalGeneration(config)
        pretrained_decoder = pretrained.model.language_model
        model_class = Gemma3TextVLForConditionalGeneration
    else:
        upstream_class, model_class = {
            "qwen3": (Qwen3ForCausalLM, Qwen3TextVLForConditionalGeneration),
            "phi4": (Phi3ForCausalLM, Phi4MiniVLForConditionalGeneration),
        }[backbone]
        pretrained = upstream_class(text_config)
        pretrained_decoder = pretrained.model
    pretrained.save_pretrained(tmp_path)

    model = model_class.from_pretrained(tmp_path, local_files_only=True)

    embeddings = model.get_input_embeddings().weight
    assert model.lm_head.weight.data_ptr() == embeddings.data_ptr()
    loaded = model.model.language_model.state_dict()
    for name, value in pretrained_decoder.state_dict().items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_trainable_parameters_keep_names_and_dtypes(monkeypatch, tmp_path, tie_word_embeddings):
    monkeypatch.setenv("VLM_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VLM_OUTPUT_DIR", str(tmp_path))
    from vlm.configs.model.trainable_params.trainable_linear_layer import (
        PEFTLoraLinearConfig,
        TrainableLinearConfig,
    )
    from vlm.configs.model.trainable_params.trainable_module import (
        PEFTLoraConfig,
        TrainableModuleConfig,
    )
    from vlm.models.qwen3.wrapper import Qwen3TextVLWrapper

    # The backbone is loaded in bfloat16; all trainable parameters are kept in float32. The base
    # layer of the lm_head LoRA stays frozen whether or not it is tied to the embeddings.
    config = make_config("qwen3")
    config.tie_word_embeddings = tie_word_embeddings
    model = Qwen3TextVLForConditionalGeneration(config).to(torch.bfloat16)
    model.model.vision_adapter = torch.nn.Linear(16, 16)
    wrapper = Qwen3TextVLWrapper.__new__(Qwen3TextVLWrapper)
    wrapper.model = model
    wrapper.text_only = False
    wrapper.model_cfg = SimpleNamespace(
        trainable_params=SimpleNamespace(
            vision_encoder=None,
            adapter=TrainableModuleConfig(mode="full"),
            language_model=TrainableModuleConfig(
                mode="peft", peft=PEFTLoraConfig(type="lora", r=2, target_modules=["q_proj"])
            ),
            lm_head=TrainableLinearConfig(mode="peft", peft=PEFTLoraLinearConfig(type="lora", r=2)),
        )
    )

    wrapper.configure_model_for_training()

    lora = "model.language_model.base_model.model.layers.0.self_attn.q_proj.lora_"
    assert {
        name: param.dtype for name, param in model.named_parameters() if param.requires_grad
    } == {
        "model.vision_adapter.weight": torch.float32,
        "model.vision_adapter.bias": torch.float32,
        lora + "A.default.weight": torch.float32,
        lora + "B.default.weight": torch.float32,
        "lm_head.lora_A.default.weight": torch.float32,
        "lm_head.lora_B.default.weight": torch.float32,
    }
    assert model.lm_head.base_layer.weight.dtype == torch.bfloat16
    tied = {"lm_head.base_layer.weight": "model.language_model.embed_tokens.weight"}
    assert model.all_tied_weights_keys == (tied if tie_word_embeddings else {})


def test_qwen25_save_load_and_hub_checkpoint_layout(tmp_path):
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


@pytest.mark.parametrize("processor_class", [Qwen25Processor, Qwen3Processor])
def test_processor_roundtrip_expands_custom_vision_tokens(tmp_path, processor_class):
    vocab = {"[UNK]": 0, "[PAD]": 1, "<|image_pad|>": 2, "<|video_pad|>": 3}
    tokenizer = Qwen2Tokenizer(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
        extra_special_tokens=["<|image_pad|>", "<|video_pad|>"],
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

    # Prompt-only tokenization for label masking runs without padding.
    unpadded = reloaded(
        text=["<|video_pad|>", "<|video_pad|><|video_pad|>"],
        num_vision_tokens=[2, 1, 3],
        padding=False,
    )
    assert [ids.count(3) for ids in unpadded["input_ids"]] == [2, 4]


def test_shared_registry_registers_nnfoundation_backbone(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VLM_OUTPUT_DIR", str(tmp_path))
    from vlm.models.qwen25vl.wrapper import Qwen25VLWrapper
    from vlm.registries.vlm_registry import VLM_REGISTRY

    assert VLM_REGISTRY["Qwen2.5-VL-3B"].model_class is Qwen25VLWrapper
    assert {"Qwen3-1.7B", "Qwen3-VL-2B", "Phi4Mini", "Gemma3-4B"} <= VLM_REGISTRY.keys()
    assert {"Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B"} <= VLM_REGISTRY.keys()


@pytest.mark.parametrize(
    ("eos_token_id", "expected"),
    [(None, [2]), (1, [2, 1]), ([1, 2], [1, 2])],
)
def test_generation_stops_at_tokenizer_eos(eos_token_id, expected):
    from types import SimpleNamespace

    from transformers import GenerationConfig

    from vlm.models.abstract_vl_wrapper import AbstractVLWrapper

    wrapper = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=2, pad_token_id=0),
        model=SimpleNamespace(generation_config=GenerationConfig(eos_token_id=eos_token_id)),
    )
    AbstractVLWrapper._stop_generation_at_tokenizer_eos(wrapper)
    assert wrapper.model.generation_config.eos_token_id == expected
    assert wrapper.model.generation_config.pad_token_id == 0
