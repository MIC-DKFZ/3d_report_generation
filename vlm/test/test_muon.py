"""MuonWithAdamW: parity with torch's own Muon / AdamW, and the multi-stage schedulers.

The schedulers hand the *same* param-group list to the optimizer and to the LR scheduler, which
zips its max_lrs against optimizer.param_groups by position. The Muon/AdamW split therefore has
to happen before either sees the groups (SchedulerBase._get_param_groups), and most of the
scheduler tests below pin exactly that alignment.
"""

import copy

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW, Muon

from vlm.training.optimizers.muon import MuonWithAdamW
from vlm.training.schedulers.AdapterDWUScheduler import AdapterDWUScheduler

LR, WD = 1e-2, 0.1


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(11, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8, bias=False)
        self.norm = nn.LayerNorm(8)
        self.conv = nn.Conv1d(8, 8, 3, padding=1)
        self.lm_head = nn.Linear(8, 11, bias=False)

    def forward(self, ids):
        x = self.embed_tokens(ids)                            # (B, T, 8)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(self.fc2(torch.relu(self.fc1(x))))
        return self.lm_head(x)


def _loss(model, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 11, (4, 5), generator=g)
    return nn.functional.cross_entropy(model(ids).flatten(0, 1), ids.flatten())


def _groups(model):
    return MuonWithAdamW.split_param_groups(
        [dict(name="all", params=list(model.parameters()), lr=LR, weight_decay=WD)], model
    )


def _train(model, optimizer, steps=5):
    for step in range(steps):
        optimizer.zero_grad()
        _loss(model, step).backward()
        optimizer.step()


# --------------------------------------------------------------------------- split

def test_split_routes_only_plain_matrices_to_muon():
    model = _Toy()
    names = {id(p): n for n, p in model.named_parameters()}
    groups = _groups(model)

    muon = {names[id(p)] for g in groups if g["use_muon"] for p in g["params"]}
    adamw = {names[id(p)] for g in groups if not g["use_muon"] for p in g["params"]}
    assert muon == {"fc1.weight", "fc2.weight"}
    # 1D, 3D conv kernel, and the two 2D tables left to AdamW by name.
    assert adamw == {"embed_tokens.weight", "fc1.bias", "norm.weight", "norm.bias",
                     "conv.weight", "conv.bias", "lm_head.weight"}
    assert [g["name"] for g in groups] == ["all_MUON", "all_ADAMW"]
    assert all(g["lr"] == LR and g["weight_decay"] == WD for g in groups)


def test_split_patterns_are_configurable_and_empty_halves_dropped():
    model = _Toy()
    groups = MuonWithAdamW.split_param_groups(
        [dict(name="head", params=[model.lm_head.weight], lr=LR, weight_decay=0.0)],
        model, adamw_name_patterns=[],
    )
    assert [(g["name"], g["use_muon"]) for g in groups] == [("head_MUON", True)]


def test_unsplit_groups_are_rejected():
    with pytest.raises(ValueError, match="use_muon"):
        MuonWithAdamW([dict(params=list(_Toy().parameters()))])


# --------------------------------------------------------------------------- parity

def test_muon_half_matches_torch_muon():
    ours, ref = _Toy(), _Toy()
    ref.load_state_dict(ours.state_dict())
    groups = [g for g in _groups(ours) if g["use_muon"]]
    ref_params = [dict(ref.named_parameters())[n] for n in ("fc1.weight", "fc2.weight")]
    for p in ref.parameters():  # only the Muon half moves in both models
        p.requires_grad_(any(p is q for q in ref_params))
    for p in ours.parameters():
        p.requires_grad_(any(p is q for g in groups for q in g["params"]))

    _train(ours, MuonWithAdamW(groups, lr=LR, weight_decay=WD))
    _train(ref, Muon(ref_params, lr=LR, weight_decay=WD, adjust_lr_fn="match_rms_adamw"))

    for (n, a), b in zip(ours.named_parameters(), ref.parameters()):
        torch.testing.assert_close(a, b, msg=n)


def test_adamw_half_matches_torch_adamw():
    ours, ref = _Toy(), _Toy()
    ref.load_state_dict(ours.state_dict())
    groups = [g for g in _groups(ours) if not g["use_muon"]]
    adamw_ids = {id(p) for g in groups for p in g["params"]}
    adamw_names = [n for n, p in ours.named_parameters() if id(p) in adamw_ids]
    for model, names in ((ours, adamw_names), (ref, adamw_names)):
        for n, p in model.named_parameters():
            p.requires_grad_(n in names)

    _train(ours, MuonWithAdamW(groups, lr=LR, weight_decay=WD))
    _train(ref, AdamW([p for n, p in ref.named_parameters() if n in adamw_names], lr=LR, weight_decay=WD))

    for (n, a), b in zip(ours.named_parameters(), ref.parameters()):
        torch.testing.assert_close(a, b, msg=n)


def test_state_dict_round_trip_continues_identically():
    a = _Toy()
    opt_a = MuonWithAdamW(_groups(a), lr=LR, weight_decay=WD)
    _train(a, opt_a, steps=3)

    b = copy.deepcopy(a)
    opt_b = MuonWithAdamW(_groups(b), lr=LR, weight_decay=WD)
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))

    for opt, model in ((opt_a, a), (opt_b, b)):
        opt.zero_grad()
        _loss(model, 99).backward()
        opt.step()
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, msg=n)


def test_flattened_muon_params_raise():
    """What DeepSpeed ZeRO would hand it: flat 1D shards in a Muon group."""
    flat = nn.Parameter(torch.randn(16))
    flat.grad = torch.randn(16)
    opt = MuonWithAdamW([dict(params=[flat], use_muon=True)], lr=LR)
    with pytest.raises(RuntimeError, match="ZeRO"):
        opt.step()


# --------------------------------------------------------------------------- schedulers

ADAPTER_LR, LM_LR = 4.0e-4, 2.0e-4
MAX_STEPS = 100  # -> WARMUP_1 [0, 10), WARMUP_2 [10, 15), DEFAULT from 15
STAGE_STEPS = (0, 12, 20)


class _AdapterToy(nn.Module):
    """An 'adapter' prefix (WARMUP_1 trains only it) in front of _Toy, which owns the rest."""

    def __init__(self):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 8))
        self.lm = _Toy()

    @property
    def device(self):  # SchedulerBase.restore_checkpoint_state maps the optimizer state onto it
        return next(self.parameters()).device


def _make_muon_scheduler(output_dir) -> AdapterDWUScheduler:
    return AdapterDWUScheduler(
        optimizer_class=MuonWithAdamW,
        optimizer_kwargs=dict(weight_decay=0.0),
        module_attr_map={"adapter": "adapter."},
        forbidden_name_patterns_for_wd=["embed_tokens", "lm_head"],
        output_dir=output_dir,
        adapter_lr=ADAPTER_LR,
        lm_lr=LM_LR,
        warmup_1_duration=0.1,
        warmup_2_duration=0.05,
    )


def _prepare(sched, model, step, opt, lr_sched):
    return sched.prepare_training_step(model=model, optimizer=opt, lr_scheduler=lr_sched,
                                       global_step=step, max_steps=MAX_STEPS)


def _fake_step(model, opt, lr_sched):
    for p in model.parameters():
        p.grad = torch.randn_like(p) if p.requires_grad else None
    opt.step()
    lr_sched.step()
    opt.zero_grad()


def test_every_stage_steps_with_aligned_lrs(tmp_path):
    model = _AdapterToy()
    sched = _make_muon_scheduler(tmp_path)
    sched.initialize(model=model, max_steps=MAX_STEPS)
    opt, lr_sched, optimizers, group_names = None, None, [], []
    for step in STAGE_STEPS:
        opt, lr_sched, _ = _prepare(sched, model, step, opt, lr_sched)
        optimizers.append(opt)
        group_names.append([g["name"] for g in opt.param_groups])
        assert isinstance(opt, MuonWithAdamW)
        assert len(lr_sched.max_lrs) == len(opt.param_groups)
        for group, max_lr in zip(opt.param_groups, lr_sched.max_lrs):
            module = group["name"].rsplit("_", 1)[0].removesuffix("_NO_DECAY").removesuffix("_DECAY")
            assert max_lr == {"adapter": ADAPTER_LR, "default": LM_LR}[module], group["name"]
            assert group["name"].endswith("_MUON") == group["use_muon"]
            if group["use_muon"]:
                assert all(p.ndim == 2 for p in group["params"])

        # Every trainable parameter is in exactly one group, and a step moves all of them.
        trainable = [p for p in model.parameters() if p.requires_grad]
        in_opt = [p for g in opt.param_groups for p in g["params"]]
        assert len(in_opt) == len({id(p) for p in in_opt}) == len(trainable)
        before = [p.detach().clone() for p in trainable]
        _fake_step(model, opt, lr_sched)
        assert all(not torch.equal(b, p) for b, p in zip(before, trainable))

    assert str(sched.training_stage) == "DEFAULT"
    # WARMUP_2 unfreezes the LM and rebuilds the optimizer; DEFAULT keeps it.
    assert optimizers[0] is not optimizers[1]
    assert optimizers[1] is optimizers[2]
    # WARMUP_1 holds only the adapter, split across both algorithms.
    assert group_names[0] == ["adapter_DECAY_MUON", "adapter_DECAY_ADAMW", "adapter_NO_DECAY_ADAMW"]


def test_resume_restores_the_muon_state(tmp_path):
    """SchedulerBase.resume rebuilds the checkpoint's stage, then loads its state into it."""
    from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME

    model = _AdapterToy()
    sched = _make_muon_scheduler(tmp_path)
    sched.initialize(model=model, max_steps=MAX_STEPS)
    opt, lr_sched = None, None
    for step in range(13):  # into WARMUP_2
        opt, lr_sched, _ = _prepare(sched, model, step, opt, lr_sched)
        _fake_step(model, opt, lr_sched)
    ckpt = tmp_path / "checkpoint-13"
    ckpt.mkdir()
    torch.save(opt.state_dict(), ckpt / OPTIMIZER_NAME)
    torch.save(lr_sched.state_dict(), ckpt / SCHEDULER_NAME)

    resumed_model = copy.deepcopy(model)
    for p in resumed_model.parameters():  # a fresh process starts from the configured state
        p.requires_grad_(True)
    resumed = _make_muon_scheduler(tmp_path)
    resumed.initialize(model=resumed_model, max_steps=MAX_STEPS)
    opt2, lr_sched2, _ = resumed.resume(model=resumed_model, optimizer=None, lr_scheduler=None,
                                        global_step=13, max_steps=MAX_STEPS, checkpoint_dir=str(ckpt))

    assert [g["name"] for g in opt2.param_groups] == [g["name"] for g in opt.param_groups]
    assert [g["lr"] for g in opt2.param_groups] == [g["lr"] for g in opt.param_groups]
    names = {id(p): n for n, p in model.named_parameters()}
    names2 = {id(p): n for n, p in resumed_model.named_parameters()}
    state = {names[id(p)]: s for p, s in opt.state.items()}
    state2 = {names2[id(p)]: s for p, s in opt2.state.items()}
    assert state.keys() == state2.keys() and state
    for name in state:
        for key in state[name]:
            torch.testing.assert_close(state2[name][key], state[name][key], msg=f"{name}.{key}")
