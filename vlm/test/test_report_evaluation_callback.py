"""Report callback integration, classifier failures, and saved-config migration."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from transformers import Trainer, TrainerControl, TrainerState, TrainingArguments

from vlm.configs.compatibility import normalize_saved_config
from vlm.configs.evaluation_config import EvaluationConfig
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


@pytest.mark.parametrize("training", [False, True])
def test_classifier_failure_propagates_and_restores_model_mode(tmp_path, make_callback, training):
    error = RuntimeError("classifier unavailable")

    def scorer(**kwargs):
        raise error

    callback = make_callback(scorer)
    model = TinyModel().train(training)
    args, state = args_and_state(tmp_path)
    with pytest.raises(RuntimeError, match="classifier unavailable") as exc:
        callback.on_evaluate(args, state, TrainerControl(), model=model, metrics={})
    assert exc.value is error
    assert model.training == training
    assert not (tmp_path / "callback_eval").exists()


def test_empty_classifier_result_fails_before_writing_metrics(tmp_path, make_callback):
    callback = make_callback(lambda **kwargs: ({}, None, None))
    args, state = args_and_state(tmp_path)
    with pytest.raises(ValueError, match="required 'f1'"):
        callback.on_evaluate(args, state, TrainerControl(), model=TinyModel(), metrics={})
    assert not (tmp_path / "callback_eval").exists()


def test_unknown_classifier_fails_at_construction(make_callback):
    with pytest.raises(ValueError, match="No classifier evaluator"):
        make_callback(None)


def test_explicitly_disabled_classifier(tmp_path, make_callback):
    callback = make_callback(None, calculate_classifier_score=False)
    args, state = args_and_state(tmp_path)
    metrics = {"eval_loss": 1.0}
    callback.on_evaluate(args, state, TrainerControl(), model=TinyModel(), metrics=metrics)
    assert metrics == {"eval_loss": 1.0}
    assert json.loads((tmp_path / "callback_eval/step-10/VLM_metrics.json").read_text()) == {}


def test_secondary_rank_receives_metrics_without_running_classifier(
    tmp_path, make_callback, monkeypatch
):
    def scorer(**kwargs):
        pytest.fail("Only rank zero should run the classifier")

    callback = make_callback(scorer, is_distributed=True)
    monkeypatch.setattr(evaluation, "gather_inference_outputs", lambda **kwargs: None)

    def broadcast(payload, src):
        assert src == 0
        payload[0] = {"eval_classifier_f1": 0.8}

    monkeypatch.setattr(evaluation.dist, "broadcast_object_list", broadcast)
    args, state = args_and_state(tmp_path, main=False)
    metrics = {}
    callback.on_evaluate(args, state, TrainerControl(), model=TinyModel(), metrics=metrics)
    assert metrics == {"eval_classifier_f1": 0.8}
    assert not (tmp_path / "callback_eval").exists()


@pytest.mark.parametrize("metric", ["radbert_f1", "eval_radbert_f1"])
def test_old_saved_classifier_configs_migrate_without_mutating_input(metric):
    original = {
        "eval": {"calculate_radbert_score": True},
        "trainer": {"metric_for_best_model": metric},
    }
    normalized, warnings = normalize_saved_config(original)
    assert EvaluationConfig.model_validate(normalized["eval"]).calculate_classifier_score
    assert normalized["trainer"]["metric_for_best_model"] == metric
    assert original["eval"]["calculate_radbert_score"] is True
    assert len(warnings) == 1
