"""Preserve configured trainability through scheduler warmup and resume."""

import pytest
import torch
from peft import LoraConfig, PeftModel
from torch import nn
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME

from vlm.training.schedulers.adapter_dwu_scheduler import AdapterDWUScheduler
from vlm.training.schedulers.adapter_extended_dwu_scheduler import (
    AdapterExtendedDWUScheduler,
)

MODULE_MAP = {"adapter": "adapter", "language_model": "language_model", "lm_head": "lm_head"}


def make_model():
    model = nn.Module()
    model.adapter = nn.Linear(2, 2)
    model.language_model = nn.Sequential(nn.Linear(2, 2))
    model.lm_head = nn.Linear(2, 2).requires_grad_(False)
    model.device = torch.device("cpu")
    return model


def make_scheduler(model, scheduler_class, output_dir, reset_optimizer=True):
    durations = (
        {"warmup_1_duration": 0.2, "warmup_2_duration": 0.2}
        if scheduler_class is AdapterDWUScheduler
        else {"warmup_1_1_duration": 0.1, "warmup_1_2_duration": 0.1, "warmup_2_duration": 0.2}
    )
    scheduler = scheduler_class(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.001},
        module_attr_map=MODULE_MAP,
        forbidden_name_patterns_for_wd=[],
        output_dir=output_dir,
        adapter_lr=0.001,
        lm_lr=0.001,
        reset_optimizer=reset_optimizer,
        **durations,
    )
    scheduler.initialize(model, max_steps=100)
    return scheduler


@pytest.mark.parametrize("bias", ["none", "all", "lora_only"])
def test_snapshot_preserves_peft_setup_and_reports_changes(tmp_path, bias):
    model = make_model()
    backbone = nn.Module()
    backbone.proj = nn.Linear(2, 2)
    backbone.saved = nn.Linear(2, 2)
    model.language_model = PeftModel(
        backbone, LoraConfig(r=2, target_modules=["proj"], modules_to_save=["saved"], bias=bias)
    )
    expected = {name: param.requires_grad for name, param in model.named_parameters()}
    parameter_ids = [id(param) for param in model.parameters()]
    scheduler = make_scheduler(model, AdapterDWUScheduler, tmp_path)
    for param in model.parameters():
        param.requires_grad_(not param.requires_grad)
    # Repeated initialization must not replace the pre-warmup snapshot.
    scheduler.initialize(model, max_steps=100)
    assert scheduler._restore_configured_trainable_params_state(model)
    assert {name: param.requires_grad for name, param in model.named_parameters()} == expected
    assert parameter_ids == [id(param) for param in model.parameters()]
    assert not scheduler._restore_configured_trainable_params_state(model)


@pytest.mark.parametrize("scheduler_class", [AdapterDWUScheduler, AdapterExtendedDWUScheduler])
@pytest.mark.parametrize("stage_step", [0, 10, 15, 20, 40])
def test_direct_stage_entry_and_normal_transitions(tmp_path, scheduler_class, stage_step):
    model = make_model()
    scheduler = make_scheduler(model, scheduler_class, tmp_path)
    optimizer, lr_scheduler, _ = scheduler.prepare_training_step(
        model, None, None, stage_step, 100
    )
    assert all(param.requires_grad for param in model.adapter.parameters())
    assert all(
        param.requires_grad == (stage_step >= 20) for param in model.language_model.parameters()
    )
    assert not any(param.requires_grad for param in model.lm_head.parameters())
    if stage_step < 20:
        if scheduler_class is AdapterExtendedDWUScheduler and stage_step < 10:
            optimizer, lr_scheduler, changed = scheduler.prepare_training_step(
                model, optimizer, lr_scheduler, 10, 100
            )
            assert not changed  # Both first-warmup stages are adapter-only.
        _, _, changed = scheduler.prepare_training_step(model, optimizer, lr_scheduler, 20, 100)
        assert changed
        assert all(param.requires_grad for param in model.language_model.parameters())
        assert not any(param.requires_grad for param in model.lm_head.parameters())


@pytest.mark.parametrize("scheduler_class", [AdapterDWUScheduler, AdapterExtendedDWUScheduler])
@pytest.mark.parametrize("resume_step", [1, 10, 15, 20, 25, 40, 50])
@pytest.mark.parametrize("reset_optimizer", [True, False])
def test_resume_preserves_trainability_and_optimizer_moments(
    tmp_path, scheduler_class, resume_step, reset_optimizer
):
    source = make_model()
    scheduler = make_scheduler(source, scheduler_class, tmp_path, reset_optimizer)
    optimizer, lr_scheduler, _ = scheduler.prepare_training_step(source, None, None, 0, 100)
    optimizer, lr_scheduler, _ = scheduler.prepare_training_step(
        source, optimizer, lr_scheduler, resume_step - 1, 100
    )
    loss = sum(param.square().sum() for param in source.parameters() if param.requires_grad)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    saved_moments = {
        name: optimizer.state[param]["exp_avg"].clone()
        for name, param in source.named_parameters()
        if param in optimizer.state
    }
    torch.save(optimizer.state_dict(), tmp_path / OPTIMIZER_NAME)
    torch.save(lr_scheduler.state_dict(), tmp_path / SCHEDULER_NAME)

    target = make_model()
    scheduler = make_scheduler(target, scheduler_class, tmp_path, reset_optimizer)
    optimizer, lr_scheduler, _ = scheduler.prepare_training_step(target, None, None, 0, 100)
    initial_optimizer, initial_lr_scheduler = optimizer, lr_scheduler
    optimizer, lr_scheduler, _ = scheduler.resume(target, resume_step, 100, str(tmp_path))
    assert optimizer is not initial_optimizer
    assert lr_scheduler is not initial_lr_scheduler
    assert all(
        param.requires_grad == (resume_step >= 20) for param in target.language_model.parameters()
    )
    assert all(param.requires_grad for param in target.adapter.parameters())
    assert not any(param.requires_grad for param in target.lm_head.parameters())
    if resume_step == 20 and reset_optimizer:
        assert not optimizer.state  # Preserve reset-on-second-warmup behavior.
    else:
        for name, param in target.named_parameters():
            if name in saved_moments:
                torch.testing.assert_close(optimizer.state[param]["exp_avg"], saved_moments[name])


@pytest.mark.parametrize("scheduler_class", [AdapterDWUScheduler, AdapterExtendedDWUScheduler])
@pytest.mark.parametrize("reset_optimizer", [True, False])
def test_unfreezing_preserves_state_unless_reset(tmp_path, scheduler_class, reset_optimizer):
    model = make_model()
    scheduler = make_scheduler(model, scheduler_class, tmp_path, reset_optimizer)
    optimizer, lr_scheduler, _ = scheduler.prepare_training_step(model, None, None, 0, 100)
    sum(param.square().sum() for param in model.adapter.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad()
    saved_state = {
        param: {key: value.clone() for key, value in state.items()}
        for param, state in optimizer.state.items()
    }

    optimizer, lr_scheduler, _ = scheduler.prepare_training_step(
        model, optimizer, lr_scheduler, 20, 100
    )
    assert {id(param) for group in optimizer.param_groups for param in group["params"]} == {
        id(param) for param in model.parameters() if param.requires_grad
    }
    if reset_optimizer:
        assert not optimizer.state
    else:
        for param, state in saved_state.items():
            for key, value in state.items():
                torch.testing.assert_close(optimizer.state[param][key], value)

    # Retained parameters continue their step count; newly unfrozen ones start fresh.
    sum(param.square().sum() for param in model.parameters() if param.requires_grad).backward()
    optimizer.step()
    for param in model.adapter.parameters():
        assert optimizer.state[param]["step"].item() == (1 if reset_optimizer else 2)
    for param in model.language_model.parameters():
        assert optimizer.state[param]["step"].item() == 1


@pytest.mark.parametrize("scheduler_class", [AdapterDWUScheduler, AdapterExtendedDWUScheduler])
def test_second_warmup_starts_at_first_warmup_step_lr(tmp_path, scheduler_class):
    model = make_model()
    scheduler = make_scheduler(model, scheduler_class, tmp_path)
    optimizer, lr_scheduler, _ = scheduler.prepare_training_step(model, None, None, 0, 100)
    optimizer, lr_scheduler, _ = scheduler.prepare_training_step(
        model, optimizer, lr_scheduler, 20, 100
    )
    # With gradient accumulation, accelerate skips the explicit step(global_step) on the
    # first micro-batch, so the LR set at construction is used for the stage's first step.
    expected = [0.001 / 20] * len(optimizer.param_groups)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(expected)
    lr_scheduler.step(20)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(expected)


def test_optimizer_detects_membership_change_with_same_group_count(tmp_path):
    model = make_model()
    scheduler = make_scheduler(model, AdapterDWUScheduler, tmp_path)
    optimizer, _ = scheduler._configure_optimizer(model, None)
    reused_optimizer, _ = scheduler._configure_optimizer(model, optimizer)
    assert reused_optimizer is optimizer

    # Replace a parameter while retaining exactly the same number and sizes of groups.
    old_weight = model.adapter.weight
    model.adapter.weight = nn.Parameter(old_weight.detach().clone())
    updated_optimizer, _ = scheduler._configure_optimizer(model, optimizer)
    assert updated_optimizer is not optimizer
    assert len(updated_optimizer.param_groups) == len(optimizer.param_groups)
    parameter_ids = {
        id(param) for group in updated_optimizer.param_groups for param in group["params"]
    }
    assert id(model.adapter.weight) in parameter_ids
    assert id(old_weight) not in parameter_ids
