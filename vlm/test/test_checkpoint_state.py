"""Small CPU tensor save/load tests; no trainers or training runs."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from vlm.training.checkpoint_state import (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    checkpoint_state_dict,
    ensure_checkpoint_state,
    initialize_checkpoint_state,
    load_model_checkpoint,
    read_manifest,
    record_trainable_checkpoint_keys,
    save_model_checkpoint,
)


def make_model():
    model = nn.Module()
    model.base = nn.Linear(2, 2, bias=False).requires_grad_(False)
    model.adapter = nn.Linear(2, 2, bias=False)
    model.register_buffer("counter", torch.zeros(1))
    initialize_checkpoint_state(model)
    return model


def test_partial_roundtrip_preserves_frozen_required_weights(tmp_path):
    source = make_model()
    with torch.no_grad():
        source.adapter.weight.fill_(7)
        source.counter.fill_(3)
    source.adapter.requires_grad_(False)
    save_model_checkpoint(source, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {WEIGHTS_NAME, MANIFEST_NAME}
    assert read_manifest(tmp_path) == {"previous_run_dirs": []}

    target = make_model()
    original_base = target.base.weight.clone()
    load_model_checkpoint(target, tmp_path)
    torch.testing.assert_close(target.adapter.weight, source.adapter.weight)
    torch.testing.assert_close(target.counter, source.counter)
    torch.testing.assert_close(target.base.weight, original_base)


def test_loaded_weights_remain_cumulative_after_freezing(tmp_path):
    source = make_model()
    source.base.requires_grad_(True)
    save_model_checkpoint(source, tmp_path / "first")
    target = make_model()
    load_model_checkpoint(target, tmp_path / "first")
    target.requires_grad_(False)
    save_model_checkpoint(target, tmp_path / "second")
    assert "base.weight" in load_file(tmp_path / "second" / WEIGHTS_NAME)


def test_newly_trainable_keys_remain_required():
    model = make_model()
    model.base.requires_grad_(True)
    record_trainable_checkpoint_keys(model)
    model.base.requires_grad_(False)
    assert "base.weight" in checkpoint_state_dict(model)


def test_missing_required_key_on_save(tmp_path):
    model = make_model()
    ensure_checkpoint_state(model).required_keys.add("missing.weight")
    with pytest.raises(ValueError, match="Required checkpoint keys missing"):
        save_model_checkpoint(model, tmp_path)


@pytest.mark.parametrize("value", [torch.zeros(3, 3), torch.zeros(2, 2, dtype=torch.float64)])
def test_shape_and_dtype_validation(tmp_path, value):
    save_file({"adapter.weight": value}, tmp_path / WEIGHTS_NAME)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"previous_run_dirs": []}))
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        load_model_checkpoint(make_model(), tmp_path)


def test_loaded_key_must_exist_in_model(tmp_path):
    save_file({"unknown.weight": torch.zeros(2, 2)}, tmp_path / WEIGHTS_NAME)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"previous_run_dirs": []}))
    with pytest.raises(ValueError, match="does not exist in model"):
        load_model_checkpoint(make_model(), tmp_path)


def test_legacy_load_marks_loaded_weights_required(tmp_path):
    source = make_model()
    checkpoint_dir = tmp_path / "legacy_run" / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True)
    save_file({"base.weight": source.base.weight.detach().clone()}, checkpoint_dir / WEIGHTS_NAME)
    target = make_model()
    load_model_checkpoint(target, checkpoint_dir)
    torch.testing.assert_close(target.base.weight, source.base.weight)
    target.requires_grad_(False)
    save_model_checkpoint(target, tmp_path / "next")
    assert "base.weight" in load_file(tmp_path / "next" / WEIGHTS_NAME)
    assert read_manifest(tmp_path / "next") == {
        "previous_run_dirs": [str(tmp_path / "legacy_run")]
    }


def test_previous_run_directories_are_carried_forward(tmp_path):
    first_checkpoint = tmp_path / "sft" / "checkpoint-1"
    save_model_checkpoint(make_model(), first_checkpoint)

    second = make_model()
    load_model_checkpoint(second, first_checkpoint)
    second_checkpoint = tmp_path / "dpo" / "checkpoint-1"
    save_model_checkpoint(second, second_checkpoint)
    assert read_manifest(second_checkpoint) == {"previous_run_dirs": [str(tmp_path / "sft")]}

    third = make_model()
    load_model_checkpoint(third, second_checkpoint)
    assert ensure_checkpoint_state(third).previous_run_dirs == [
        str(tmp_path / "sft"),
        str(tmp_path / "dpo"),
    ]


def test_shared_parameters_are_saved_without_deduplication(tmp_path):
    model = make_model()
    model.head = nn.Linear(2, 2, bias=False)
    model.head.weight = model.adapter.weight
    save_model_checkpoint(model, tmp_path)
    weights = load_file(tmp_path / WEIGHTS_NAME)
    assert "head.weight" in weights and "adapter.weight" in weights
    torch.testing.assert_close(weights["head.weight"], weights["adapter.weight"])
