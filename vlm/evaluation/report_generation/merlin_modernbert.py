"""Evaluate generated MERLIN reports using ModernBERT finding labels."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from vlm.evaluation.report_generation.utils import compute_label_metrics, get_reference_labels


MODEL_ID = "MIC-DKFZ/ModernBERT-Merlin"
MAX_LENGTH = 1024
THRESHOLD = 0.5


class MerlinDataset(Dataset):
    """Tokenizes reports for inference (returns input_ids + attention_mask)."""

    def __init__(self, data_files, class_count, label_cols, hf_model_dir=None, max_length=MAX_LENGTH):
        self.data_files = data_files
        self.class_count = class_count
        self.label_cols = label_cols
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        report = self.data_files["reports"][idx]
        encodings = self.tokenizer(
            report,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return encodings["input_ids"], encodings["attention_mask"]


class MerlinBertClassifier(torch.nn.Module):
    """Thin wrapper around the fine-tuned ModernBERT sequence-classification model."""

    def __init__(self, n_classes, hf_model_dir):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_dir, dtype=torch.float32
        )
        if self.model.config.num_labels != n_classes:
            raise ValueError(
                f"MERLIN checkpoint has {self.model.config.num_labels} labels, expected {n_classes}"
            )

    def forward(self, input_ids, attn_mask):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)
        return self.model(input_ids=input_ids, attention_mask=attn_mask).logits


class MerlinBertInference:
    """Runs the classifier over a dataloader and returns binary predictions."""

    def __init__(self, model, dataloaders, num_class, test_label_cols, threshold=THRESHOLD):
        self.model = model
        self.dataloaders = dataloaders
        self.num_class = num_class
        self.threshold = threshold
        self.test_label_cols = test_label_cols

    def infer(self):
        self.model.eval()
        device = next(self.model.parameters()).device
        pred_logits = np.zeros(self.num_class).reshape(1, self.num_class)

        with torch.no_grad():
            for input_ids, attention_mask in self.dataloaders["test"]:
                input_ids = torch.squeeze(input_ids, dim=1).to(device)
                attention_mask = torch.squeeze(attention_mask, dim=1).to(device)
                logits = self.model(input_ids, attention_mask)
                pred_logits = np.concatenate(
                    (pred_logits, logits.detach().float().cpu().numpy()), axis=0
                )

        pred_logits = pred_logits[1:]
        pred_labels = expit(pred_logits)
        pred_labels[pred_labels >= self.threshold] = 1
        pred_labels[pred_labels < self.threshold] = 0
        return pred_labels


def load_label_columns(model_id: str | Path | None = None) -> list[str]:
    """Read finding names in classifier-output order from the model config."""
    model_id = MODEL_ID if model_id is None else str(model_id)
    config = AutoConfig.from_pretrained(model_id)
    mapping = config.id2label
    if not mapping or set(mapping) != set(range(config.num_labels)):
        raise ValueError("MERLIN config.id2label must map every output index to a finding")
    labels = [mapping[index] for index in range(config.num_labels)]
    return labels


def predict_labels(reports, model_id: str | Path | None = None):
    """Classify reports with the fine-tuned Merlin ModernBERT model -> [N, 30] binary df.

    The model source may be a Hugging Face repository ID or a local checkpoint.
    """

    def _as_report_list(reports):
        """Accept either a list of strings or the pycocoeval dict {i: [text]}."""
        if isinstance(reports, dict):
            return [
                reports[k][0] if isinstance(reports[k], (list, tuple)) else reports[k]
                for k in sorted(reports)
            ]
        return list(reports)

    reports = _as_report_list(reports)
    model_id = MODEL_ID if model_id is None else str(model_id)
    label_cols = load_label_columns(model_id)
    df = pd.DataFrame({"reports": reports})

    batch_size = 32
    if os.getenv("SLURM_JOB_ID") or os.getenv("LSB_JOBID"):
        batch_size = 96

    ds = MerlinDataset(df, len(label_cols), label_cols, hf_model_dir=model_id)
    loader = DataLoader(
        ds, sampler=SequentialSampler(ds), batch_size=batch_size, num_workers=4, pin_memory=True
    )

    model = MerlinBertClassifier(n_classes=len(label_cols), hf_model_dir=model_id)
    model = model.cuda().eval()

    trainer = MerlinBertInference(model, {"test": loader}, len(label_cols), label_cols)
    preds = trainer.infer()  # ndarray [N, C]
    return pd.DataFrame(preds.astype(int), columns=label_cols)


def evaluate_reports(
    preds,
    gts,
    image_ids: list[str],
    default_gt_df: pd.DataFrame | None = None,
    model_id: str | Path | None = None,
    gt_labels_path: str | Path | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Compare ModernBERT predictions against supplied or classified references.

    Returns (aggregate_metrics, pred_df, gt_df); the dfs are [N, 30] binary
    finding labels. References come from a supplied dataframe or explicit CSV,
    then gt_labels.csv in the local model directory or pinned Hub snapshot.
    Otherwise ModernBERT classifies the reference reports.
    """
    if not (len(preds) == len(gts) == len(image_ids)):
        raise ValueError(
            "Classifier evaluation inputs must have equal lengths: "
            f"preds={len(preds)}, gts={len(gts)}, image_ids={len(image_ids)}"
        )
    model_id = MODEL_ID if model_id is None else str(model_id)
    label_cols = load_label_columns(model_id)
    gt_df = get_reference_labels(
        gts,
        image_ids,
        label_cols,
        lambda reports: predict_labels(reports, model_id),
        default_gt_df=default_gt_df,
        gt_labels_path=gt_labels_path,
        model_id=model_id,
    )
    pred_df = predict_labels(preds, model_id)

    results = compute_label_metrics(
        pred_df,
        gt_df,
        label_columns=label_cols,
        zero_division=0,
    )
    return results, pred_df, gt_df
