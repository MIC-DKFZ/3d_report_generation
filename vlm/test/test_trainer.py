"""VLMTrainer hooks and trainer configuration."""

import inspect

from transformers import Trainer

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
