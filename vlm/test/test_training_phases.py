"""Phased training: one Trainer per phase, resumable from any checkpoint."""

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from accelerate.state import AcceleratorState
from safetensors.torch import load_file
from test_gradient_accumulation import (
    SequentialTrainer,
    collate,
    make_model,
    make_samples,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, TrainerCallback, TrainingArguments

from vlm.training.checkpoint_state import (
    initialize_checkpoint_state,
    load_model_checkpoint,
)
from vlm.training.phases import train_phases
from vlm.training.schedulers.adapter_dwu_scheduler import AdapterDWUScheduler

# Four epochs of four optimizer steps. WARMUP_1 (epoch 0): steps 0-3, adapter only.
# DEFAULT (epochs 1-3): second warmup in steps 4-7, then cosine decay.
EPOCHS, SAMPLES_PER_EPOCH, ACCUMULATION = 4, 8, 2
MAX_STEPS = EPOCHS * SAMPLES_PER_EPOCH // ACCUMULATION
# The decoder layers act as the adapter, so embeddings stay frozen until DEFAULT.
MODULE_ATTR_MAP = {"adapter": "model.language_model.layers"}
EMBEDDINGS = "model.language_model.embed_tokens.weight"


class OptimizerRecorder(TrainerCallback):
    """Record the learning rates and the AdamW step count before every optimizer step."""

    def __init__(self):
        self.steps = {}

    def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        # Muon keeps no step count, only a momentum buffer.
        adam_steps = [int(param_state.get("step", 0)) for param_state in optimizer.state.values()]
        self.steps[state.global_step] = (
            [group["lr"] for group in optimizer.param_groups],
            max(adam_steps, default=0),
        )


class StopAtStep(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_training_stop = True


def train(
    output_dir,
    checkpoint=None,
    world_size=1,
    callbacks=(),
    save_strategy="steps",
    deepspeed=None,
    bf16=False,
    tie_word_embeddings=False,
    stop_requested=lambda: False,
    optimizer_class=torch.optim.AdamW,
    optimizer_kwargs=None,
):
    model = make_model(tie_word_embeddings)
    initialize_checkpoint_state(model)
    if checkpoint is not None:
        load_model_checkpoint(model, checkpoint)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "a": 2, "b": 3})),
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    scheduler = AdapterDWUScheduler(
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs or {"weight_decay": 0.0},
        module_attr_map=MODULE_ATTR_MAP,
        forbidden_name_patterns_for_wd=[],
        output_dir=output_dir,
        adapter_lr=0.01,
        lm_lr=0.005,
        warmup_1_duration=0.25,
        warmup_2_duration=0.25,
    )
    recorder = OptimizerRecorder()
    args = TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=deepspeed is None,
        bf16=bf16,
        deepspeed=deepspeed,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=ACCUMULATION,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to=[],
        save_strategy=save_strategy,
        save_steps=1,
        logging_strategy="no",
        dataloader_pin_memory=False,
        disable_tqdm=True,
    )

    def create_trainer(phase, phase_start_state):
        scheduler.set_trainable_params(model, phase)
        return SequentialTrainer(
            model=model,
            args=args,
            processing_class=tokenizer,
            data_collator=collate,
            train_dataset=make_samples(SAMPLES_PER_EPOCH * world_size),
            callbacks=[recorder, *callbacks],
            scheduler=scheduler,
            phase=phase,
            phase_start_state=phase_start_state,
            model_weights_preloaded=checkpoint is not None,
        )

    trained = train_phases(scheduler.phases(EPOCHS), create_trainer, checkpoint, stop_requested)
    return model, trained, recorder.steps


def flat_parameters(model):
    return torch.cat([param.detach().flatten() for param in model.parameters()])


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("uninterrupted")
    initial = make_model().state_dict()[EMBEDDINGS]
    model, trained, steps = train(output_dir)
    assert trained
    return output_dir, initial, flat_parameters(model), steps


def test_phases_train_their_parameters_with_a_fresh_optimizer(uninterrupted):
    output_dir, initial_embeddings, _, steps = uninterrupted
    assert sorted(steps) == list(range(MAX_STEPS))
    # One adapter group in WARMUP_1, plus the language model group in DEFAULT.
    assert [len(steps[step][0]) for step in (3, 4)] == [1, 2]
    # AdamW counts steps per parameter: the DEFAULT optimizer starts again at zero.
    assert [steps[step][1] for step in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]

    boundary = load_file(output_dir / "checkpoint-4" / "model.safetensors")
    final = load_file(output_dir / f"checkpoint-{MAX_STEPS}" / "model.safetensors")
    torch.testing.assert_close(boundary[EMBEDDINGS], initial_embeddings, rtol=0, atol=0)
    assert not torch.equal(final[EMBEDDINGS], initial_embeddings)


def test_learning_rates_follow_the_schedule(uninterrupted):
    *_, steps = uninterrupted
    lr = AdapterDWUScheduler(
        torch.optim.AdamW, {}, MODULE_ATTR_MAP, [], "", 0.01, 0.005, 0.25, 0.25
    ).lr_schedule(MAX_STEPS, MAX_STEPS // EPOCHS, EPOCHS)
    for step, (learning_rates, _) in steps.items():
        max_lrs = [0.01] if step < 4 else [0.01, 0.005]
        assert learning_rates == pytest.approx([lr(step, max_lr) for max_lr in max_lrs])


# Inside WARMUP_1, at the phase boundary, and inside DEFAULT.
@pytest.mark.parametrize("resume_step", [2, 4, 6])
def test_resume_matches_uninterrupted_training(tmp_path, uninterrupted, resume_step):
    output_dir, _, expected_parameters, expected_steps = uninterrupted
    model, trained, steps = train(tmp_path, output_dir / f"checkpoint-{resume_step}")

    assert trained
    assert steps == {step: value for step, value in expected_steps.items() if step >= resume_step}
    torch.testing.assert_close(flat_parameters(model), expected_parameters)


def test_resume_from_final_checkpoint_takes_no_step(tmp_path, uninterrupted):
    output_dir, _, expected_parameters, _ = uninterrupted
    model, trained, steps = train(tmp_path, output_dir / f"checkpoint-{MAX_STEPS}")

    assert not trained
    assert not steps
    torch.testing.assert_close(flat_parameters(model), expected_parameters)


def test_phase_end_is_saved_without_checkpointing(tmp_path):
    train(tmp_path, save_strategy="no")
    # A later phase or a resumed run starts from this checkpoint.
    assert [path.name for path in tmp_path.glob("checkpoint-*")] == ["checkpoint-4"]


def test_stopping_early_ends_the_run_instead_of_the_phase(tmp_path):
    _, trained, steps = train(tmp_path, callbacks=[StopAtStep(2)])
    assert trained
    assert sorted(steps) == [0, 1]


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
        model, _, _ = train(output_dir, world_size=world_size)
        # Embeddings only train in the second phase; without a new DDP wrapper, their
        # gradients would not be synchronized and the ranks would diverge.
        assert not torch.equal(model.get_input_embeddings().weight, initial_embeddings)
        parameters = flat_parameters(model)
        gathered = [torch.empty_like(parameters) for _ in range(world_size)]
        dist.all_gather(gathered, parameters)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU DDP requires Gloo")
def test_two_process_ddp_phase_change_keeps_ranks_in_sync(tmp_path):
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
