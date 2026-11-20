"""MuonWithAdamW: parity with torch's own Muon and AdamW, per-phase groups, and resumes."""


import pytest
import torch
import torch.nn as nn
from test_training_phases import flat_parameters, train
from torch.optim import AdamW, Muon

from vlm.training.optimizers.muon import MuonWithAdamW

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


@pytest.fixture(scope="module")
def muon_uninterrupted(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("muon_uninterrupted")
    model, trained, steps = train(output_dir, optimizer_class=MuonWithAdamW)
    return output_dir, flat_parameters(model), steps


# Inside WARMUP_1, at the phase boundary, and inside DEFAULT.
