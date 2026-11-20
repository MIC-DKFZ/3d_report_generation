"""Small CPU tensor save/load tests; no trainers or training runs."""


import torch
from torch import nn

from vlm.training.checkpoint_state import (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    initialize_checkpoint_state,
    load_model_checkpoint,
    read_manifest,
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
