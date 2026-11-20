"""Report callback integration, classifier failures, and saved-config migration."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from transformers import Trainer, TrainerControl, TrainerState, TrainingArguments

from vlm.registries import evaluation_registry
from vlm.training.callbacks import ReportEvaluationCallback
from vlm.training.callbacks import evaluation


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, labels=None, **kwargs):
        return {"loss": self.weight.square(), "logits": input_ids.float().unsqueeze(-1)}

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.full((len(input_ids), 1), 4)], dim=1)


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "reference"

    def batch_decode(self, ids, **kwargs):
        return ["generated\nreport"] * len(ids)


def generation_batch(image_id):
    return {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "input_ids_prompt_only": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[-100, -100, 3]]),
        "raw_samples": {"image_id": [image_id]},
        "batch_count": 1,
    }


@pytest.fixture
def make_callback(monkeypatch):
    monkeypatch.setattr(evaluation.wandb, "run", None)

    def make(scorer, **kwargs):
        monkeypatch.setattr(evaluation_registry, "get_classifier_evaluator", lambda _: scorer)
        monkeypatch.setattr(evaluation_registry, "load_classifier", lambda _: None)
        return ReportEvaluationCallback(
            [generation_batch("first"), generation_batch("second")],
            max_new_tokens=2,
            dataset_name="merlin",
            tokenizer=Tokenizer(),
            **kwargs,
        )

    return make


def args_and_state(tmp_path, *, main=True):
    return (
        SimpleNamespace(device=torch.device("cpu"), output_dir=str(tmp_path)),
        TrainerState(epoch=2, global_step=10, is_world_process_zero=main),
    )


def test_trainer_evaluation_adds_classifier_metric_before_best_model_selection(
    tmp_path, make_callback
):
    calls = []
    labels = pd.DataFrame({"finding": [1, 0]})

    def scorer(**kwargs):
        calls.append(kwargs)
        return {"f1": 0.75, "precision": 0.8}, labels, labels

    callback = make_callback(scorer)
    trainer = Trainer(
        model=TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            disable_tqdm=True,
            dataloader_pin_memory=False,
            metric_for_best_model="eval_classifier_f1",
            greater_is_better=True,
        ),
        # Trainer loss evaluation has one sample; callback generation has two.
        eval_dataset=[{"input_ids": [1, 2], "labels": [1, 2]}],
        callbacks=[callback],
    )
    trainer.state.epoch = 2
    trainer.state.global_step = 10
    metrics = trainer.evaluate()

    assert metrics["eval_classifier_f1"] == 0.75
    assert trainer.state.log_history[-1]["eval_classifier_f1"] == 0.75
    assert trainer._determine_best_metric(metrics, trial=None)
    assert trainer.state.best_metric == 0.75
    assert calls[0]["image_ids"] == ["first", "second"]
    assert calls[0]["preds"] == ["generated report"] * 2
    assert calls[0]["default_gt_df"] is None
    folder = tmp_path / "callback_eval/step-10"
    assert json.loads((folder / "VLM_metrics.json").read_text())["f1"] == 0.75
    assert list(pd.read_csv(folder / "classifier_predictions.csv")["image_id"]) == [
        "first",
        "second",
    ]
    assert (folder.parent / "f1_score.png").is_file()

    args, state = args_and_state(tmp_path)
    callback.on_evaluate(args, state, TrainerControl(), model=TinyModel(), metrics={})
    assert calls[1]["default_gt_df"] is labels
