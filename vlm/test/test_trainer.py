"""VLMTrainer hooks and trainer configuration."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from transformers import Trainer

from vlm.configs.compatibility import normalize_saved_config
from vlm.configs.trainer_config import TrainerConfig
from vlm.training.checkpoint_state import save_model_checkpoint
from vlm.training.trainers import VLMTrainer


def parameters(method):
    return [
        (param.name, param.kind, param.default)
        for param in inspect.signature(method).parameters.values()
    ]


def test_trainer_overrides_match_upstream_methods():
    # Upstream only calls an override while the hook keeps its name and parameters,
    # so a renamed or changed private hook would otherwise be skipped silently.
    for name, method in vars(VLMTrainer).items():
        if name.startswith("__") or not callable(method):
            continue
        upstream = getattr(Trainer, name, None)
        assert upstream is not None, f"Trainer no longer defines '{name}'"
        assert parameters(method) == parameters(upstream), f"Trainer.{name} changed"


@pytest.mark.parametrize(
    "field, value",
    [("deepspeed", "config.json"), ("dataset_kwargs", {"skip_prepare_dataset": True})],
)
def test_removed_trainer_field_is_rejected_but_old_saved_configs_load(field, value):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainerConfig(**{field: value})
    normalized, warnings = normalize_saved_config({"trainer": {field: value}})
    assert TrainerConfig.model_validate(normalized["trainer"])
    assert any(f"trainer.{field}" in message for message in warnings)


@pytest.mark.parametrize("preloaded", [False, True])
def test_checkpoint_loading_respects_preloaded_weights(tmp_path, preloaded):
    source = torch.nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(7)
    save_model_checkpoint(source, tmp_path)
    target = torch.nn.Linear(2, 2)
    original = target.weight.detach().clone()
    trainer = SimpleNamespace(model=target, model_weights_preloaded=preloaded)

    VLMTrainer._load_from_checkpoint(trainer, tmp_path)

    torch.testing.assert_close(target.weight, original if preloaded else source.weight)
