"""Staged warmup training must survive DDP re-wrapping and resuming from any checkpoint."""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from accelerate.state import AcceleratorState
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, TrainerCallback
from trl import SFTConfig

from test_gradient_accumulation import SequentialTrainer, collate, make_model, make_samples
from vlm.training.checkpoint_state import initialize_checkpoint_state, load_model_checkpoint
from vlm.training.schedulers.adapter_dwu_scheduler import AdapterDWUScheduler

MAX_STEPS = 8  # WARMUP_1: steps 0-1, WARMUP_2: steps 2-3, DEFAULT: steps 4-7
# The decoder layers act as the adapter, so embeddings stay frozen until WARMUP_2.
MODULE_ATTR_MAP = {"adapter": "model.language_model.layers"}


class LearningRateRecorder(TrainerCallback):
    def __init__(self):
        self.learning_rates = {}

    def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        self.learning_rates[state.global_step] = [group["lr"] for group in optimizer.param_groups]


def make_trainer(model, output_dir, samples, recorder, checkpoint=None):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "a": 2, "b": 3})),
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    scheduler = AdapterDWUScheduler(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"weight_decay": 0.0},
        module_attr_map=MODULE_ATTR_MAP,
        forbidden_name_patterns_for_wd=[],
        output_dir=output_dir,
        adapter_lr=0.01,
        lm_lr=0.005,
        warmup_1_duration=0.25,
        warmup_2_duration=0.25,
    )
    return SequentialTrainer(
        model=model,
        processing_class=tokenizer,
        scheduler=scheduler,
        optimizer_scheduler_checkpoint=checkpoint,
        model_weights_preloaded=checkpoint is not None,
        data_collator=collate,
        train_dataset=Dataset.from_list(samples),
        callbacks=[recorder],
        args=SFTConfig(
            output_dir=str(output_dir),
            use_cpu=True,
            bf16=False,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            report_to=[],
            save_strategy="steps",
            save_steps=1,
            logging_strategy="no",
            dataloader_pin_memory=False,
            disable_tqdm=True,
        ),
    )


def train(output_dir, samples, checkpoint=None):
    model = make_model()
    initialize_checkpoint_state(model)
    if checkpoint is not None:
        load_model_checkpoint(model, checkpoint)
    recorder = LearningRateRecorder()
    trainer = make_trainer(model, output_dir, samples, recorder, checkpoint)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint is not None else None)
    return model, trainer, recorder.learning_rates


def flat_parameters(model):
    return torch.cat([param.detach().flatten() for param in model.parameters()])


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("uninterrupted")
    model, _, learning_rates = train(output_dir, make_samples(2 * MAX_STEPS))
    return output_dir, flat_parameters(model), learning_rates


@pytest.mark.parametrize("resume_step", [1, 2, 3, 5])
def test_resume_matches_uninterrupted_training(tmp_path, uninterrupted, resume_step):
    output_dir, expected_parameters, expected_learning_rates = uninterrupted
    checkpoint = output_dir / f"checkpoint-{resume_step}"
    model, trainer, learning_rates = train(tmp_path, make_samples(2 * MAX_STEPS), checkpoint)

    assert trainer.state.global_step == MAX_STEPS
    assert learning_rates == {
        step: rates for step, rates in expected_learning_rates.items() if step >= resume_step
    }
    torch.testing.assert_close(flat_parameters(model), expected_parameters)


def test_resume_from_final_checkpoint_takes_no_step(tmp_path, uninterrupted):
    output_dir, expected_parameters, _ = uninterrupted
    checkpoint = output_dir / f"checkpoint-{MAX_STEPS}"
    model, trainer, learning_rates = train(tmp_path, make_samples(2 * MAX_STEPS), checkpoint)

    assert trainer.already_complete_on_resume
    assert trainer.state.global_step == MAX_STEPS
    assert not learning_rates
    torch.testing.assert_close(flat_parameters(model), expected_parameters)


def ddp_worker(rank, world_size, rendezvous, output_dir):
    torch.set_num_threads(1)
    AcceleratorState._reset_state(reset_partial_state=True)
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29501",
    )
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=Path(rendezvous).as_uri(),
        timeout=timedelta(seconds=60),
    )
    try:
        initial_embeddings = make_model().get_input_embeddings().weight.detach().clone()
        model, _, _ = train(output_dir, make_samples(2 * MAX_STEPS * world_size))
        # Embeddings are unfrozen after DDP was built; without re-wrapping, their
        # gradients are not synchronized and the ranks diverge.
        assert not torch.equal(model.get_input_embeddings().weight, initial_embeddings)
        parameters = flat_parameters(model)
        gathered = [torch.empty_like(parameters) for _ in range(world_size)]
        dist.all_gather(gathered, parameters)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU DDP requires Gloo")
def test_two_process_ddp_stage_transition_keeps_ranks_in_sync(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.multiprocessing.start_processes(
            ddp_worker,
            args=(2, str(tmp_path / "rendezvous"), str(tmp_path)),
            nprocs=2,
            start_method="spawn",
        )
    finally:
        torch.set_num_threads(previous_threads)
