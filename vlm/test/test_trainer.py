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
from vlm.training.trainers import VLMTrainer, checkpointed_layer_indices


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


def test_removed_trainer_field_is_rejected_but_old_saved_configs_load():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainerConfig(dataset_kwargs={"skip_prepare_dataset": True})
    normalized, warnings = normalize_saved_config(
        {"trainer": {"dataset_kwargs": {"skip_prepare_dataset": True}}}
    )
    assert TrainerConfig.model_validate(normalized["trainer"])
    assert any("trainer.dataset_kwargs" in message for message in warnings)


def test_deepspeed_config_path_keeps_its_expression(monkeypatch):
    monkeypatch.setenv("VLM_REPO", "/repo")
    config = TrainerConfig(deepspeed="${VLM_REPO}/paper_configs/deepspeed/zero2.json")
    assert config.model_dump()["deepspeed"] == "${VLM_REPO}/paper_configs/deepspeed/zero2.json"
    assert config.deepspeed.get_resolved_path() == "/repo/paper_configs/deepspeed/zero2.json"


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


@pytest.mark.parametrize(
    "num_layers, fraction, expected",
    [
        (4, 1.0, [0, 1, 2, 3]),
        (4, 0.5, [1, 3]),
        (4, 0.25, [3]),
        (4, 0.1, []),
        (28, 0.7, None),
    ],
)
def test_checkpointed_layers_are_the_fraction_spread_evenly(num_layers, fraction, expected):
    indices = checkpointed_layer_indices(num_layers, fraction)
    if expected is not None:
        assert indices == expected
    assert len(indices) == round(fraction * num_layers)
    # Spread evenly: the gaps between checkpointed layers differ by at most one.
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    assert max(gaps, default=0) - min(gaps, default=0) <= 1


def test_gradient_checkpointing_fraction_is_validated():
    assert TrainerConfig(
        gradient_checkpointing=True, gradient_checkpointing_fraction=0.7
    ).gradient_checkpointing_fraction == 0.7
    with pytest.raises(ValidationError, match="requires 'gradient_checkpointing'"):
        TrainerConfig(gradient_checkpointing_fraction=0.7)
    with pytest.raises(ValidationError, match="not both"):
        TrainerConfig(
            gradient_checkpointing=True,
            gradient_checkpointing_fraction=0.7,
            gradient_checkpointing_kwargs={"use_reentrant": False, "every_n_layers": 2},
        )
    for fraction in (0, 1.5):
        with pytest.raises(ValidationError):
            TrainerConfig(gradient_checkpointing=True, gradient_checkpointing_fraction=fraction)


def test_only_the_checkpointed_layers_recompute_in_the_backward(tmp_path):
    from test_gradient_accumulation import make_model, make_samples, make_trainer

    model = make_model(num_hidden_layers=4)
    layers = model.model.language_model.layers
    forward_calls = [0] * len(layers)
    for index, layer in enumerate(layers):
        # A pre-hook, since the recompute stops once it has the tensors the backward needs,
        # before the layer's forward returns.
        layer.register_forward_pre_hook(
            lambda *_, index=index: forward_calls.__setitem__(index, forward_calls[index] + 1)
        )

    make_trainer(
        model, make_samples(1), 1, 1, tmp_path, gradient_checkpointing_fraction=0.5
    ).train()

    assert [layer.gradient_checkpointing for layer in layers] == [False, True, False, True]
    # One training step: a checkpointed layer runs its forward again during the backward.
    assert forward_calls == [1, 2, 1, 2]
