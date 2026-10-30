"""Scheduler phases, per-phase trainability, and learning-rate schedules."""

import pytest
import torch
from peft import LoraConfig, PeftModel
from torch import nn

from vlm.training.schedulers.adapter_dwu_scheduler import AdapterDWUScheduler
from vlm.training.schedulers.adapter_extended_dwu_scheduler import (
    AdapterExtendedDWUScheduler,
)
from vlm.training.schedulers.base import MIN_LR, TrainingPhase

MODULE_MAP = {"adapter": "adapter", "language_model": "language_model", "lm_head": "lm_head"}
ADAPTER_ONLY = TrainingPhase("WARMUP_1", 0, 1, adapter_only=True)
DEFAULT = TrainingPhase("DEFAULT", 1, 2)


def make_model():
    model = nn.Module()
    model.adapter = nn.Linear(2, 2)
    model.language_model = nn.Sequential(nn.Linear(2, 2))
    model.lm_head = nn.Linear(2, 2).requires_grad_(False)
    return model


def make_scheduler(scheduler_class=AdapterDWUScheduler, output_dir="", **durations):
    if not durations:
        durations = (
            {"warmup_1_duration": 0.2, "warmup_2_duration": 0.1}
            if scheduler_class is AdapterDWUScheduler
            else {"warmup_1_1_duration": 0.1, "warmup_1_2_duration": 0.1, "warmup_2_duration": 0.1}
        )
    return scheduler_class(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={},
        module_attr_map=MODULE_MAP,
        forbidden_name_patterns_for_wd=[],
        output_dir=output_dir,
        adapter_lr=0.001,
        lm_lr=0.002,
        **durations,
    )


def trainable(model):
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_adapter_only_phase_freezes_everything_but_the_adapter():
    model = make_model()
    scheduler = make_scheduler()
    scheduler.set_trainable_params(model, ADAPTER_ONLY)
    assert trainable(model) == {"adapter.weight", "adapter.bias"}
    scheduler.set_trainable_params(model, DEFAULT)
    # The lm_head stays frozen as configured.
    assert trainable(model) == {"adapter.weight", "adapter.bias"} | {
        "language_model.0.weight",
        "language_model.0.bias",
    }


@pytest.mark.parametrize("bias", ["none", "all", "lora_only"])
def test_default_phase_restores_the_configured_peft_setup(bias):
    model = make_model()
    backbone = nn.Module()
    backbone.proj = nn.Linear(2, 2)
    backbone.saved = nn.Linear(2, 2)
    model.language_model = PeftModel(
        backbone, LoraConfig(r=2, target_modules=["proj"], modules_to_save=["saved"], bias=bias)
    )
    configured = trainable(model)
    parameter_ids = [id(param) for param in model.parameters()]
    scheduler = make_scheduler()

    scheduler.set_trainable_params(model, ADAPTER_ONLY)
    scheduler.set_trainable_params(model, DEFAULT)

    assert trainable(model) == configured
    assert parameter_ids == [id(param) for param in model.parameters()]


def test_each_phase_optimizer_contains_exactly_its_trainable_parameters(tmp_path):
    model = make_model()
    scheduler = make_scheduler(output_dir=tmp_path)
    for phase, expected_lrs in [(ADAPTER_ONLY, {0.001}), (DEFAULT, {0.001, 0.002})]:
        scheduler.set_trainable_params(model, phase)
        optimizer = scheduler.create_optimizer(model, phase)
        assert not optimizer.state
        assert {id(param) for group in optimizer.param_groups for param in group["params"]} == {
            id(param) for param in model.parameters() if param.requires_grad
        }
        assert {group["lr"] for group in optimizer.param_groups} == expected_lrs
        assert (tmp_path / f"param_groups__{phase.name}.log").is_file()


@pytest.mark.parametrize(
    "scheduler_class, durations, num_train_epochs, warmup_1_epochs",
    [
        (AdapterDWUScheduler, {"warmup_1_duration": 0.1}, 12, 1),
        (AdapterDWUScheduler, {"warmup_1_duration": 0.1}, 24, 2),
        (AdapterDWUScheduler, {"warmup_1_duration": 0.0625}, 32, 2),
        (AdapterDWUScheduler, {"warmup_1_duration": 0.0}, 4, 0),
        (
            AdapterExtendedDWUScheduler,
            {"warmup_1_1_duration": 0.0625, "warmup_1_2_duration": 0.1875},
            16,
            4,
        ),
    ],
)
def test_adapter_only_phase_is_rounded_to_whole_epochs(
    scheduler_class, durations, num_train_epochs, warmup_1_epochs
):
    scheduler = make_scheduler(scheduler_class, warmup_2_duration=0.05, **durations)
    phases = scheduler.phases(num_train_epochs)
    default = TrainingPhase("DEFAULT", warmup_1_epochs, num_train_epochs)
    if warmup_1_epochs:
        assert phases == [
            TrainingPhase("WARMUP_1", 0, warmup_1_epochs, adapter_only=True),
            default,
        ]
    else:
        assert phases == [default]


@pytest.mark.parametrize("warmup_1_duration", [0.01, 0.95])
def test_adapter_only_phase_must_round_to_a_proper_part_of_training(warmup_1_duration):
    scheduler = make_scheduler(warmup_1_duration=warmup_1_duration, warmup_2_duration=0.0)
    with pytest.raises(ValueError, match="at least one epoch"):
        scheduler.phases(10)


# 10 epochs of 10 steps. WARMUP_1: steps 0-19, second warmup: 20-29, cosine decay: 30-99.
@pytest.mark.parametrize(
    "scheduler_class, step, expected",
    [
        (AdapterDWUScheduler, 0, 0.05),
        (AdapterDWUScheduler, 9, 0.5),
        (AdapterDWUScheduler, 19, 1.0),
        (AdapterExtendedDWUScheduler, 0, 0.1),
        (AdapterExtendedDWUScheduler, 9, 1.0),
        (AdapterExtendedDWUScheduler, 19, 1.0),
    ],
)
def test_first_warmup_learning_rates(scheduler_class, step, expected):
    lr = make_scheduler(scheduler_class).lr_schedule(100, 10, 10)
    assert lr(step, 1.0) == pytest.approx(expected)


@pytest.mark.parametrize("scheduler_class", [AdapterDWUScheduler, AdapterExtendedDWUScheduler])
@pytest.mark.parametrize(
    "step, expected",
    [(20, 0.1), (29, 1.0), (30, 1.0), (65, MIN_LR + (1.0 - MIN_LR) / 2), (99, 0.0005)],
)
def test_second_warmup_and_decay_learning_rates(scheduler_class, step, expected):
    lr = make_scheduler(scheduler_class).lr_schedule(100, 10, 10)
    assert lr(step, 1.0) == pytest.approx(expected, abs=1e-4)


def test_lr_scheduler_starting_later_continues_the_schedule():
    model = make_model()
    scheduler = make_scheduler()
    lr = scheduler.lr_schedule(100, 10, 10)
    optimizer = torch.optim.SGD([{"params": model.adapter.parameters(), "lr": 0.001}])
    lr_scheduler = scheduler.create_lr_scheduler(
        optimizer, max_steps=100, steps_per_epoch=10, num_train_epochs=10, start_step=25
    )
    for step in range(25, 30):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(lr(step, 0.001))
        optimizer.step()
        lr_scheduler.step()
