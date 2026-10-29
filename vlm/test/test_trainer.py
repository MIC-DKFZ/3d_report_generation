"""Small CPU training run through the custom scheduler's stage transition."""

import inspect
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from vlm.configs.compatibility import normalize_saved_config
from vlm.configs.trainer_config import TrainerConfig
from vlm.training.checkpoint_state import save_model_checkpoint
from vlm.training.trainers import VLMTrainer

# Methods of VLMTrainer that do not override a Trainer method.
OWN_TRAINER_METHODS = {"_prepare_training_step"}


def parameters(method):
    return [
        (param.name, param.kind, param.default)
        for param in inspect.signature(method).parameters.values()
    ]


def test_trainer_overrides_match_upstream_methods():
    # Upstream only calls an override while the hook keeps its name and parameters,
    # so a renamed or changed private hook would otherwise be skipped silently.
    for name, method in vars(VLMTrainer).items():
        if name.startswith("__") or not callable(method) or name in OWN_TRAINER_METHODS:
            continue
        upstream = getattr(Trainer, name, None)
        assert upstream is not None, f"Trainer no longer defines '{name}'"
        assert parameters(method) == parameters(upstream), f"Trainer.{name} changed"


@pytest.mark.parametrize(
    "field, value",
    [("deepspeed", "config.json"), ("dataset_kwargs", {"skip_prepare_dataset": True})],
)
def test_removed_trainer_field_is_rejected_but_old_saved_configs_load(field, value):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainerConfig(**{field: value})
    normalized, warnings = normalize_saved_config({"trainer": {field: value}})
    assert TrainerConfig.model_validate(normalized["trainer"])
    assert any(f"trainer.{field}" in message for message in warnings)


@pytest.mark.parametrize("preloaded", [False, True])
def test_checkpoint_loading_respects_preloaded_weights(tmp_path, preloaded):
    source = torch.nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(7)
    save_model_checkpoint(source, tmp_path)
    target = torch.nn.Linear(2, 2)
    original = target.weight.detach().clone()
    trainer = SimpleNamespace(model=target, model_weights_preloaded=preloaded)

    VLMTrainer._load_from_checkpoint(trainer, tmp_path)

    torch.testing.assert_close(target.weight, original if preloaded else source.weight)


def test_best_checkpoint_restores_weights(tmp_path):
    source = torch.nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(7)
    save_model_checkpoint(source, tmp_path)
    target = torch.nn.Linear(2, 2)
    trainer = SimpleNamespace(
        model=target,
        state=SimpleNamespace(best_model_checkpoint=tmp_path),
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
    )

    VLMTrainer._load_best_model(trainer)

    torch.testing.assert_close(target.weight, source.weight)


def collate_causal_lm(samples):
    input_ids = torch.tensor([sample["input_ids"] for sample in samples])
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def test_cpu_training_with_accumulation_and_stage_transition(tmp_path):
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=8, n_embd=8, n_layer=1, n_head=1))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "a": 2, "b": 3})),
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    parameter = model.transformer.h[0].attn.c_attn.weight
    initial_weight = parameter.detach().clone()

    class StageScheduler:
        frozen_weight = None

        def initialize(self, model, max_steps):
            pass

        def prepare_training_step(self, model, optimizer, lr_scheduler, global_step, max_steps):
            if optimizer is None:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            changed = global_step == 1 and self.frozen_weight is None
            if changed:
                parameter.requires_grad_(False)
                self.frozen_weight = parameter.detach().clone()
            return optimizer, lr_scheduler, changed

    scheduler = StageScheduler()
    trainer = VLMTrainer(
        model=model,
        processing_class=tokenizer,
        scheduler=scheduler,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            bf16=False,
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            remove_unused_columns=False,
            report_to=[],
            save_strategy="no",
            logging_strategy="no",
            dataloader_pin_memory=False,
            disable_tqdm=True,
        ),
        train_dataset=[{"input_ids": [2, 3, 2, 1]}] * 4,
        data_collator=collate_causal_lm,
    )

    result = trainer.train()

    assert result.global_step == 2
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert scheduler.frozen_weight is not None
    assert not torch.equal(initial_weight, scheduler.frozen_weight)
    torch.testing.assert_close(parameter, scheduler.frozen_weight)
