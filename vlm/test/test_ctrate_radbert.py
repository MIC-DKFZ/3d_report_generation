"""Exercise Hugging Face classifier loading and batching without network or CUDA."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from vlm.evaluation.report_generation import ctrate_radbert


class FakeClassifier(torch.nn.Module):
    def __init__(self, num_labels=18):
        super().__init__()
        self.config = SimpleNamespace(num_labels=num_labels)
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        assert input_ids.ndim == attention_mask.ndim == 2
        # First input token encodes a negative, zero, or positive test logit.
        logits = input_ids[:, :1].float().expand(-1, self.config.num_labels)
        return SimpleNamespace(logits=logits)


def test_hf_classifier_predicts_in_order_including_singleton_last_batch(monkeypatch):
    model_loader = Mock(return_value=FakeClassifier())
    tokenizer = Mock(
        side_effect=lambda report, **kwargs: {
            "input_ids": torch.tensor([[int(report), 1, 2]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }
    )
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(
        ctrate_radbert.AutoModelForSequenceClassification, "from_pretrained", model_loader
    )
    monkeypatch.setattr(ctrate_radbert.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(ctrate_radbert.RadBertClassifier, "cuda", lambda self: self)
    monkeypatch.setattr(
        ctrate_radbert,
        "DataLoader",
        lambda dataset, sampler, **kwargs: DataLoader(dataset, sampler=sampler, batch_size=2),
    )

    labels = ctrate_radbert.predict_labels(["1", "-1", "0"])

    pd.testing.assert_frame_equal(
        labels,
        pd.DataFrame(
            np.repeat([[1], [0], [1]], len(ctrate_radbert.LABEL_COLUMNS), axis=1),
            columns=ctrate_radbert.LABEL_COLUMNS,
        ),
    )
    model_loader.assert_called_once_with(ctrate_radbert.MODEL_ID, dtype=torch.float32)
    tokenizer_loader.assert_called_once_with(ctrate_radbert.MODEL_ID, do_lower_case=True)
    for call in tokenizer.call_args_list:
        assert call.kwargs == {
            "return_tensors": "pt",
            "max_length": 512,
            "padding": "max_length",
            "truncation": True,
        }


def test_hf_classifier_rejects_wrong_number_of_findings(monkeypatch):
    monkeypatch.setattr(
        ctrate_radbert.AutoModelForSequenceClassification,
        "from_pretrained",
        Mock(return_value=FakeClassifier(num_labels=2)),
    )

    with pytest.raises(ValueError, match="2 labels, expected 18"):
        ctrate_radbert.RadBertClassifier()


def test_supplied_reference_labels_skip_reference_inference(monkeypatch):
    labels = pd.DataFrame({column: [1, 0] for column in ctrate_radbert.LABEL_COLUMNS})
    predictor = Mock(return_value=labels)
    monkeypatch.setattr(ctrate_radbert, "predict_labels", predictor)

    _, _, references = ctrate_radbert.evaluate_reports(
        ["pred 1", "pred 2"], ["ref 1", "ref 2"], ["1", "2"], default_gt_df=labels
    )

    predictor.assert_called_once_with(["pred 1", "pred 2"])
    pd.testing.assert_frame_equal(references, labels)
