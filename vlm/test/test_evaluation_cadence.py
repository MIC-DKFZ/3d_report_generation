"""Epoch cadence is independent of the Trainer's optimizer-step interval."""

import pytest
import torch
from pydantic import ValidationError
from transformers import Trainer, TrainerCallback, TrainingArguments

from vlm.configs.evaluation_config import EvaluationConfig
from vlm.training.callbacks import EveryNEpochsCallback


@pytest.mark.parametrize("interval", [0, -1, 0.5, 1.5])
def test_epoch_interval_must_be_a_positive_whole_number(interval):
    with pytest.raises(ValidationError):
        EvaluationConfig(eval_every_n_epochs=interval)
    with pytest.raises(ValueError):
        EveryNEpochsCallback(n=interval)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, labels=None):
        return {"loss": self.weight.square(), "logits": input_ids.float().unsqueeze(-1)}


class RecordSteps(TrainerCallback):
    def __init__(self):
        self.evaluated_steps = []
        self.saved_steps = []

    def on_evaluate(self, args, state, control, **kwargs):
        self.evaluated_steps.append(state.global_step)

    def on_save(self, args, state, control, **kwargs):
        self.saved_steps.append(state.global_step)


def train_five_epochs(tmp_path, strategy, save_strategy="epoch") -> RecordSteps:
    """Train five two-step epochs, evaluating every second epoch or every three steps."""
    record = RecordSteps()
    callbacks = [record]
    if strategy == "epoch":
        callbacks.append(EveryNEpochsCallback(n=2))
    trainer = Trainer(
        model=Model(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            disable_tqdm=True,
            dataloader_pin_memory=False,
            num_train_epochs=5,
            per_device_train_batch_size=1,
            eval_strategy=strategy,
            eval_steps=3,
            save_strategy=save_strategy,
            logging_strategy="no",
        ),
        train_dataset=[{"input_ids": [1, 2], "labels": [1, 2]}] * 2,
        eval_dataset=[{"input_ids": [1, 2], "labels": [1, 2]}],
        callbacks=callbacks,
    )
    trainer.train()
    return record


@pytest.mark.parametrize("strategy,expected_steps", [("epoch", [4, 8, 10]), ("steps", [3, 6, 9, 10])])
def test_trainer_evaluates_at_requested_cadence(tmp_path, strategy, expected_steps):
    assert train_five_epochs(tmp_path, strategy).evaluated_steps == expected_steps


@pytest.mark.parametrize("save_strategy,expected_steps", [("epoch", [2, 4, 6, 8, 10]), ("no", [4, 8, 10])])
def test_epochs_between_evaluations_are_saved_as_the_save_strategy_says(tmp_path, save_strategy, expected_steps):
    assert train_five_epochs(tmp_path, "epoch", save_strategy).saved_steps == expected_steps

