"""Phased training and resuming under DeepSpeed ZeRO on one GPU."""

import os

import pytest
import torch
from safetensors.torch import load_file
from test_gradient_accumulation import make_model
from test_training_phases import EMBEDDINGS, MAX_STEPS, train
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="DeepSpeed requires CUDA")


def zero_config(stage):
    return {
        "bf16": {"enabled": "auto"},
        "zero_optimization": {"stage": stage, "stage3_gather_16bit_weights_on_model_save": True},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    }


def train_worker(rank, stage, bf16, port, output_dir, checkpoint, result_path):
    os.environ.update(
        RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)
    )
    _, _, steps = train(output_dir, checkpoint, deepspeed=zero_config(stage), bf16=bf16)
    unset_hf_deepspeed_config()
    torch.save(steps, result_path)


def run(tmp_path, stage, bf16, name, checkpoint=None):
    """Train in a fresh process, since DeepSpeed and accelerate keep global state."""
    output_dir = tmp_path / name
    port = 29600 + 10 * stage + len(list(tmp_path.iterdir()))
    torch.multiprocessing.start_processes(
        train_worker,
        args=(stage, bf16, port, str(output_dir), checkpoint, str(tmp_path / f"{name}.pt")),
        nprocs=1,
        start_method="spawn",
    )
    steps = torch.load(tmp_path / f"{name}.pt")
    weights = load_file(output_dir / f"checkpoint-{MAX_STEPS}" / "model.safetensors")
    return output_dir, steps, weights


@pytest.mark.parametrize("stage", [2, 3])
@pytest.mark.parametrize("bf16", [False, True])
def test_phases_and_resume_match_uninterrupted_training(tmp_path, stage, bf16):
    output_dir, expected_steps, expected_weights = run(tmp_path, stage, bf16, "uninterrupted")
    assert sorted(expected_steps) == list(range(MAX_STEPS))
    # One adapter group in WARMUP_1, plus the language model group in DEFAULT.
    assert [len(expected_steps[step][0]) for step in (3, 4)] == [1, 2]
    boundary = load_file(output_dir / "checkpoint-4" / "model.safetensors")
    initial = make_model().state_dict()[EMBEDDINGS].to(boundary[EMBEDDINGS].dtype)
    torch.testing.assert_close(boundary[EMBEDDINGS], initial, rtol=0, atol=0)
    assert not torch.equal(expected_weights[EMBEDDINGS], boundary[EMBEDDINGS])
    # DeepSpeed checkpoints leave out the frozen weights, which come from the pretrained model.
    (model_states,) = (output_dir / "checkpoint-2").glob("global_step*/*model_states.pt")
    module_states = torch.load(model_states, weights_only=False)
    assert EMBEDDINGS not in (module_states["module"] or {})
    assert not module_states.get("frozen_param_fragments")

    for resume_step in (2, 4, 6):
        checkpoint = str(output_dir / f"checkpoint-{resume_step}")
        _, steps, weights = run(tmp_path, stage, bf16, f"resume_{resume_step}", checkpoint)
        assert steps == {
            step: value for step, value in expected_steps.items() if step >= resume_step
        }
        assert weights.keys() == expected_weights.keys()
        for name, value in weights.items():
            torch.testing.assert_close(value, expected_weights[name], msg=name)
