"""MuonWithAdamW: parity with torch's own Muon and AdamW, per-phase groups, and resumes."""

import copy

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel
from test_trainability import ADAPTER_ONLY, DEFAULT, MODULE_MAP, make_model
from test_training_phases import flat_parameters, train
from torch.optim import AdamW, Muon

from vlm.registries.optimizer_registry import OPTIMIZER_REGISTRY
from vlm.training.optimizers.muon import MuonWithAdamW
from vlm.training.schedulers.adapter_dwu_scheduler import AdapterDWUScheduler

LR, WD = 1e-2, 0.1


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(11, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8, bias=False)
        self.norm = nn.LayerNorm(8)
        self.conv = nn.Conv1d(8, 8, 3, padding=1)
        self.lm_head = nn.Linear(8, 11, bias=False)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(self.fc2(torch.relu(self.fc1(x))))
        return self.lm_head(x)


def loss(model, seed):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 11, (4, 5), generator=generator)
    return nn.functional.cross_entropy(model(ids).flatten(0, 1), ids.flatten())


def split(model, **kwargs):
    return MuonWithAdamW.split_param_groups(
        [dict(name="all", params=list(model.parameters()), lr=LR, weight_decay=WD)], model, **kwargs
    )


def run(model, optimizer, steps=5):
    for step in range(steps):
        optimizer.zero_grad()
        loss(model, step).backward()
        optimizer.step()


def names_by_algorithm(model, groups):
    names = {id(param): name for name, param in model.named_parameters()}
    return (
        {names[id(p)] for group in groups if group["use_muon"] for p in group["params"]},
        {names[id(p)] for group in groups if not group["use_muon"] for p in group["params"]},
    )


def test_muon_is_a_registered_optimizer():
    assert MuonWithAdamW in OPTIMIZER_REGISTRY


def test_split_routes_only_plain_matrices_to_muon():
    model = Toy()
    groups = split(model)
    muon, adamw = names_by_algorithm(model, groups)
    assert muon == {"fc1.weight", "fc2.weight"}
    # 1D tensors, the 3D convolution kernel, and the two 2D tables left to AdamW by name.
    assert adamw == {
        "embed_tokens.weight", "fc1.bias", "norm.weight", "norm.bias",
        "conv.weight", "conv.bias", "lm_head.weight",
    }
    assert [group["name"] for group in groups] == ["all_MUON", "all_ADAMW"]
    assert all(group["lr"] == LR and group["weight_decay"] == WD for group in groups)


def test_lora_matrices_stay_on_adamw_by_name():
    backbone = nn.Module()
    backbone.proj = nn.Linear(8, 8)
    model = nn.Module()
    model.adapter = nn.Linear(8, 8)
    model.language_model = PeftModel(backbone, LoraConfig(r=2, target_modules=["proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    groups = MuonWithAdamW.split_param_groups(
        [dict(name="all", params=params, lr=LR, weight_decay=0.0)],
        model,
        adamw_name_patterns=["embed", "lm_head", "lora_"],
    )
    muon, adamw = names_by_algorithm(model, groups)
    assert muon == {"adapter.weight"}
    assert {name for name in adamw if "lora_" in name} == {
        "language_model.base_model.model.proj.lora_A.default.weight",
        "language_model.base_model.model.proj.lora_B.default.weight",
    }


def test_empty_halves_are_dropped():
    model = Toy()
    groups = MuonWithAdamW.split_param_groups(
        [dict(name="head", params=[model.lm_head.weight], lr=LR, weight_decay=0.0)],
        model,
        adamw_name_patterns=[],
    )
    assert [(group["name"], group["use_muon"]) for group in groups] == [("head_MUON", True)]


def test_unsplit_groups_are_rejected():
    with pytest.raises(ValueError, match="use_muon"):
        MuonWithAdamW([dict(params=list(Toy().parameters()))])


@pytest.mark.parametrize("use_muon", [True, False])
def test_each_half_matches_torch(use_muon):
    ours, reference = Toy(), Toy()
    reference.load_state_dict(ours.state_dict())
    groups = [group for group in split(ours) if group["use_muon"] == use_muon]
    moved = {id(p) for group in groups for p in group["params"]}
    moved_names = {name for name, p in ours.named_parameters() if id(p) in moved}
    for model in (ours, reference):
        for name, param in model.named_parameters():
            param.requires_grad_(name in moved_names)
    reference_params = [p for name, p in reference.named_parameters() if name in moved_names]

    run(ours, MuonWithAdamW(groups, lr=LR, weight_decay=WD))
    if use_muon:
        reference_optimizer = Muon(
            reference_params, lr=LR, weight_decay=WD, adjust_lr_fn="match_rms_adamw"
        )
    else:
        reference_optimizer = AdamW(reference_params, lr=LR, weight_decay=WD)
    run(reference, reference_optimizer)

    for (name, a), b in zip(ours.named_parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, msg=name)


def test_state_dict_round_trip_continues_identically():
    a = Toy()
    optimizer_a = MuonWithAdamW(split(a), lr=LR, weight_decay=WD)
    run(a, optimizer_a, steps=3)
    b = copy.deepcopy(a)
    optimizer_b = MuonWithAdamW(split(b), lr=LR, weight_decay=WD)
    optimizer_b.load_state_dict(copy.deepcopy(optimizer_a.state_dict()))

    for optimizer, model in ((optimizer_a, a), (optimizer_b, b)):
        optimizer.zero_grad()
        loss(model, 99).backward()
        optimizer.step()
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, msg=name)


def test_flattened_muon_params_raise():
    """DeepSpeed ZeRO would hand the optimizer flat 1D shards."""
    flat = nn.Parameter(torch.randn(16))
    flat.grad = torch.randn(16)
    optimizer = MuonWithAdamW([dict(params=[flat], use_muon=True)], lr=LR)
    with pytest.raises(RuntimeError, match="ZeRO"):
        optimizer.step()


def test_each_phase_splits_its_groups_and_keeps_their_learning_rates(tmp_path):
    scheduler = AdapterDWUScheduler(
        optimizer_class=MuonWithAdamW,
        optimizer_kwargs={},
        module_attr_map=MODULE_MAP,
        forbidden_name_patterns_for_wd=[],
        output_dir=tmp_path,
        adapter_lr=0.001,
        lm_lr=0.002,
        warmup_1_duration=0.2,
        warmup_2_duration=0.1,
    )
    model = make_model()
    for phase, expected in [
        (ADAPTER_ONLY, {"adapter_DECAY_MUON": 0.001, "adapter_DECAY_ADAMW": 0.001}),
        (
            DEFAULT,
            {
                "adapter_DECAY_MUON": 0.001,
                "adapter_DECAY_ADAMW": 0.001,
                "default_DECAY_MUON": 0.002,
                "default_DECAY_ADAMW": 0.002,
            },
        ),
    ]:
        scheduler.set_trainable_params(model, phase)
        optimizer = scheduler.create_optimizer(model, phase)
        assert isinstance(optimizer, MuonWithAdamW)
        assert {group["name"]: group["lr"] for group in optimizer.param_groups} == expected
        assert all(
            p.ndim == 2 for group in optimizer.param_groups if group["use_muon"] for p in group["params"]
        )
        lr_scheduler = scheduler.create_lr_scheduler(
            optimizer, max_steps=50, steps_per_epoch=5, num_train_epochs=10, start_step=0
        )
        assert len(lr_scheduler.lr_lambdas) == len(optimizer.param_groups)


@pytest.fixture(scope="module")
def muon_uninterrupted(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("muon_uninterrupted")
    model, trained, steps = train(output_dir, optimizer_class=MuonWithAdamW)
    return output_dir, flat_parameters(model), steps


def test_phased_training_uses_muon_for_the_matrices(muon_uninterrupted):
    output_dir, _, _ = muon_uninterrupted
    log = (output_dir / "param_groups__DEFAULT.log").read_text()
    assert "Algorithm: Muon" in log and "Algorithm: AdamW" in log
    # The tiny model's embeddings and output head are 2D but stay on AdamW by name.
    muon_part = log.split("Algorithm: Muon")[1].split("=== Parameter Group")[0]
    assert "embed_tokens" not in muon_part and "lm_head" not in muon_part


# Inside WARMUP_1, at the phase boundary, and inside DEFAULT.
@pytest.mark.parametrize("resume_step", [2, 4, 6])
def test_resume_matches_uninterrupted_training(tmp_path, muon_uninterrupted, resume_step):
    output_dir, expected_parameters, expected_steps = muon_uninterrupted
    model, trained, steps = train(
        tmp_path, output_dir / f"checkpoint-{resume_step}", optimizer_class=MuonWithAdamW
    )
    assert trained
    assert steps == {step: value for step, value in expected_steps.items() if step >= resume_step}
    torch.testing.assert_close(flat_parameters(model), expected_parameters)
