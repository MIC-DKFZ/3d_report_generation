"""Accumulated updates must match a single batch of supervised tokens."""

import copy
import os
import runpy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.state import AcceleratorState
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import SequentialSampler
from transformers import PreTrainedTokenizerFast, Qwen3Config, TrainingArguments

from vlm.configs.trainer_config import TrainerConfig
from vlm.training.schedulers.base import TrainingPhase
from vlm.training.trainers import VLMTrainer


class SequentialTrainer(VLMTrainer):
    def _get_train_sampler(self, *args):
        return SequentialSampler(self.train_dataset)


class ConstantScheduler:
    def create_optimizer(self, model, phase):
        return torch.optim.SGD(model.parameters(), lr=0.1)

    def create_lr_scheduler(self, optimizer, **kwargs):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def make_model():
    # Load the model implementation without importing the application wrapper/registry.
    path = Path(__file__).parents[1] / "models/qwen3/modeling_qwen3_text_vl.py"
    model_class = runpy.run_path(str(path))["Qwen3TextVLForConditionalGeneration"]
    torch.manual_seed(17)
    return model_class(
        Qwen3Config(
            vocab_size=4,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
            attention_dropout=0,
        )
    )


def make_samples(count):
    samples = []
    for index in range(count):
        prompt_length = index % 3
        # With four DDP ranks, this also gives each rank a different token total.
        input_ids = [2] * prompt_length + [2, 3] * (index % 4 + 1) + [1]
        samples.append(
            {"input_ids": input_ids, "labels": [-100] * prompt_length + input_ids[prompt_length:]}
        )
    return samples


def collate(samples):
    input_ids = [torch.tensor(sample["input_ids"]) for sample in samples]
    labels = [torch.tensor(sample["labels"]) for sample in samples]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": input_ids.ne(0).long()}


def make_trainer(model, samples, batch_size, accumulation_steps, output_dir):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "a": 2, "b": 3})),
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    return SequentialTrainer(
        model=model,
        processing_class=tokenizer,
        scheduler=ConstantScheduler(),
        phase=TrainingPhase("DEFAULT", 0, 1),
        data_collator=collate,
        train_dataset=samples,
        args=TrainingArguments(
            output_dir=str(output_dir),
            use_cpu=True,
            bf16=False,
            max_steps=1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation_steps,
            average_tokens_across_devices=TrainerConfig().average_tokens_across_devices,
            max_grad_norm=0,
            remove_unused_columns=False,
            report_to=[],
            save_strategy="no",
            logging_strategy="no",
            dataloader_pin_memory=False,
            disable_tqdm=True,
        ),
    )


def train_once(model, samples, batch_size, accumulation_steps, output_dir):
    trainer = make_trainer(model, samples, batch_size, accumulation_steps, output_dir)
    assert trainer.model_accepts_loss_kwargs
    result = trainer.train()
    assert result.global_step == 1
    return result.training_loss


def reference_update(model, samples):
    inputs = collate(samples)
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 4), inputs["labels"][:, 1:].reshape(-1))
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    return loss.item()


def assert_same_weights(actual, expected):
    for actual_param, expected_param in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(actual_param, expected_param, atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("sample_count", [5, 6])
def test_accumulation_matches_token_mean(tmp_path, sample_count):
    samples = make_samples(sample_count)
    base = make_model()
    reference = copy.deepcopy(base)
    expected_loss = reference_update(reference, samples)
    for batch_size, accumulation_steps in [(1, 6), (3, 2), (6, 1)]:
        model = copy.deepcopy(base)
        loss = train_once(model, samples, batch_size, accumulation_steps, tmp_path)
        assert loss == pytest.approx(expected_loss, rel=1e-6)
        assert_same_weights(model, reference)


def test_token_count_excludes_first_label_prompt_and_padding(tmp_path):
    trainer = make_trainer(make_model(), make_samples(1), 1, 1, tmp_path)
    batch = {"labels": torch.tensor([[2, -100, 3, 1], [-100, 2, 1, -100]])}
    batches, count = trainer.get_batch_samples(iter([batch]), 6, torch.device("cpu"))
    assert batches == [batch]
    assert count.item() == 4


def test_empty_iterator_and_unsupervised_window(tmp_path):
    trainer = make_trainer(make_model(), make_samples(1), 1, 1, tmp_path)
    assert trainer.get_batch_samples(iter([]), 6, torch.device("cpu")) == ([], None)
    batch = {"labels": torch.tensor([[2, -100]])}
    with pytest.raises(ValueError, match="no supervised tokens"):
        trainer.get_batch_samples(iter([batch]), 6, torch.device("cpu"))


def ddp_worker(rank, world_size, rendezvous, output_dir):
    torch.set_num_threads(1)
    AcceleratorState._reset_state(reset_partial_state=True)
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
    )
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=Path(rendezvous).as_uri(),
        timeout=timedelta(seconds=60),
    )
    try:
        samples = make_samples(24)
        base = make_model()
        reference = copy.deepcopy(base)
        expected_loss = reference_update(reference, samples)
        for batch_size, accumulation_steps in [(1, 6), (3, 2)]:
            model = copy.deepcopy(base)
            loss = train_once(model, samples, batch_size, accumulation_steps, output_dir)
            # TrainOutput contains the local loss when step logging is disabled.
            mean_loss = torch.tensor(loss)
            dist.all_reduce(mean_loss)
            assert mean_loss.item() / world_size == pytest.approx(expected_loss, rel=1e-6)
            assert_same_weights(model, reference)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU DDP requires Gloo")
def test_four_process_ddp_matches_global_token_mean(tmp_path):
    # Fresh workers avoid inheriting initialized autograd and thread pools.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.multiprocessing.start_processes(
            ddp_worker,
            args=(4, str(tmp_path / "rendezvous"), str(tmp_path)),
            nprocs=4,
            start_method="spawn",
        )
    finally:
        torch.set_num_threads(previous_threads)
