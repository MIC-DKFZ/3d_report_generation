"""Evaluate generated MERLIN reports using ModernBERT finding labels."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from vlm.evaluation.report_generation.utils import (
    compute_label_metrics,
    load_reference_labels,
)
from vlm.utils import logger


LABEL_COLUMNS = [
    "submucosal_edema",
    "renal_hypodensities",
    "aortic_valve_calcification",
    "coronary_calcification",
    "thrombosis",
    "metastatic_disease",
    "pancreatic_atrophy",
    "renal_cyst",
    "osteopenia",
    "surgically_absent_gallbladder",
    "atelectasis",
    "abdominal_aortic_aneurysm",
    "anasarca",
    "hiatal_hernia",
    "lymphadenopathy",
    "prostatomegaly",
    "biliary_ductal_dilation",
    "cardiomegaly",
    "splenomegaly",
    "hepatomegaly",
    "atherosclerosis",
    "ascites",
    "pleural_effusion",
    "hepatic_steatosis",
    "appendicitis",
    "gallstones",
    "hydronephrosis",
    "bowel_obstruction",
    "free_air",
    "fracture",
]


class MerlinDataset(Dataset):
    """Tokenizes reports for inference (returns input_ids + attention_mask)."""

    def __init__(self, data_files, class_count, label_cols, hf_model_dir=None, max_length=1024):
        from transformers import AutoTokenizer

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

    def __init__(self, n_classes=len(LABEL_COLUMNS), hf_model_dir=None):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_dir)
        assert self.model.config.num_labels == n_classes, (
            f"checkpoint has {self.model.config.num_labels} labels, expected {n_classes}"
        )

    def forward(self, input_ids, attn_mask):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)
        return self.model(input_ids=input_ids, attention_mask=attn_mask).logits


class MerlinBertInference:
    """Runs the classifier over a dataloader and returns binary predictions."""

    def __init__(self, model, dataloaders, num_class, test_label_cols, threshold=0.5):
        self.model = model
        self.dataloaders = dataloaders
        self.num_class = num_class
        self.threshold = threshold
        self.test_label_cols = test_label_cols

    def infer(self):
        self.model.eval()
        pred_logits = np.zeros(self.num_class).reshape(1, self.num_class)

        with torch.no_grad():
            for input_ids, attention_mask in self.dataloaders["test"]:
                input_ids = torch.squeeze(input_ids, dim=1).to("cuda")
                attention_mask = torch.squeeze(attention_mask, dim=1).to("cuda")
                logits = self.model(input_ids, attention_mask)
                pred_logits = np.concatenate(
                    (pred_logits, logits.detach().float().cpu().numpy()), axis=0
                )

        pred_logits = pred_logits[1:]
        pred_labels = expit(pred_logits)
        pred_labels[pred_labels >= self.threshold] = 1
        pred_labels[pred_labels < self.threshold] = 0
        return pred_labels


def load_label_columns(model_dir):
    """Return the label order saved alongside the model (falls back to module default)."""
    f = Path(model_dir) / "label_cols.json"
    if f.exists():
        return json.loads(f.read_text())
    return LABEL_COLUMNS


def predict_labels(reports, model_dir: str | Path):
    """Classify reports with the fine-tuned Merlin ModernBERT model -> [N, 30] binary df.

    Unlike RadBERT (a custom module + ``.pth``), the Merlin model is a
    self-contained ``AutoModelForSequenceClassification`` directory, so
    ``model_dir`` points straight at it (no separate base-model / checkpoint).
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
    label_cols = load_label_columns(model_dir)
    df = pd.DataFrame({"reports": reports})

    batch_size = 32
    if os.getenv("SLURM_JOB_ID") or os.getenv("LSB_JOBID"):
        batch_size = 96

    ds = MerlinDataset(df, len(label_cols), label_cols, hf_model_dir=str(model_dir))
    loader = DataLoader(
        ds, sampler=SequentialSampler(ds), batch_size=batch_size, num_workers=4, pin_memory=True
    )

    model = MerlinBertClassifier(n_classes=len(label_cols), hf_model_dir=str(model_dir))
    model = model.cuda().eval()

    trainer = MerlinBertInference(model, {"test": loader}, len(label_cols), label_cols)
    preds = trainer.infer()  # ndarray [N, C]
    return pd.DataFrame(preds.astype(int), columns=label_cols)


def evaluate_reports(
    preds,
    gts,
    image_ids: list[str],
    default_gt_df: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame | None, pd.DataFrame | None]:
    """Compare generated and reference reports using ModernBERT finding labels.

    Returns (aggregate_metrics, pred_df, gt_df); the dfs are [N, 30] binary
    class predictions, or None when the classifier dir is not configured.
    """
    if not (len(preds) == len(gts) == len(image_ids)):
        raise ValueError(
            "Classifier evaluation inputs must have equal lengths: "
            f"preds={len(preds)}, gts={len(gts)}, image_ids={len(image_ids)}"
        )
    model_dir = os.getenv("VLM_MERLIN_MODERNBERT_DIR")
    if model_dir is None:
        logger.warning_main(
            "The environment variable 'VLM_MERLIN_MODERNBERT_DIR' is not set. "
            "Cannot proceed with Merlin ModernBERT inference."
        )
        return {}, None, None

    model_dir = Path(model_dir)
    label_cols = load_label_columns(model_dir)
    pred_df = predict_labels(preds, model_dir)
    gt_labels_path = model_dir / "gt_labels.csv"
    if default_gt_df is not None:
        gt_df = default_gt_df.reset_index(drop=True)
    elif gt_labels_path.is_file():
        logger.info_main(f"Loading MERLIN ground-truth labels from {gt_labels_path}")
        gt_df = load_reference_labels(gt_labels_path, image_ids, label_cols)
    else:
        gt_df = predict_labels(gts, model_dir)

    results = compute_label_metrics(
        pred_df,
        gt_df,
        label_columns=LABEL_COLUMNS,
        zero_division=0,
    )
    return results, pred_df, gt_df
