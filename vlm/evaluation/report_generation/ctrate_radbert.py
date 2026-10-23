"""Evaluate generated CT-RATE reports using RadBERT finding labels."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vlm.evaluation.report_generation.utils import compute_label_metrics, get_reference_labels


MODEL_ID = "IAMJB/RadBERT-CT"
MAX_LENGTH = 512
THRESHOLD = 0.5

LABEL_COLUMNS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


class CTDataset(Dataset):
    def __init__(self, data_files, class_count, label_cols):
        self.data_files = data_files
        self.class_count = class_count

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, do_lower_case=True)
        self.max_length = MAX_LENGTH
        self.label_cols = label_cols

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

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        return input_ids, attention_mask


class RadBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, dtype=torch.float32)
        if self.model.config.num_labels != len(LABEL_COLUMNS):
            raise ValueError(
                f"CT-RATE checkpoint has {self.model.config.num_labels} labels, "
                f"expected {len(LABEL_COLUMNS)}"
            )

    def forward(self, input_ids, attn_mask):
        # ensure input_ids and attention_mask are 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)
        return self.model(input_ids=input_ids, attention_mask=attn_mask).logits


class RadBertInference:
    def __init__(
        self,
        model,
        dataloaders,
        num_class,
        test_label_cols,
        threshold=THRESHOLD,
    ):

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
                    (pred_logits, logits.detach().cpu().numpy()), axis=0
                )

        pred_logits = pred_logits[1:]
        pred_labels = expit(pred_logits)

        pred_labels[pred_labels >= self.threshold] = 1
        pred_labels[pred_labels < self.threshold] = 0

        return pred_labels


def predict_labels(reports: list[str]):
    df = pd.DataFrame({"reports": reports})
    for col in LABEL_COLUMNS:
        df[col] = 0

    batch_size = 32
    if os.getenv("SLURM_JOB_ID") or os.getenv("LSB_JOBID"):
        batch_size = 96

    # DataLoader
    ds = CTDataset(df, len(LABEL_COLUMNS), LABEL_COLUMNS)
    loader = DataLoader(
        ds, sampler=SequentialSampler(ds), batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # Model
    model = RadBertClassifier().cuda().eval()

    trainer = RadBertInference(
        model,
        {"test": loader},
        len(LABEL_COLUMNS),
        LABEL_COLUMNS,
    )

    preds = trainer.infer()  # ndarray [N, C]
    result = pd.DataFrame(preds.astype(int), columns=LABEL_COLUMNS)
    return result


def evaluate_reports(
    preds: list,
    gts: list,
    image_ids: list[str],
    default_gt_df: pd.DataFrame | None = None,
    gt_labels_path: str | Path | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Returns (aggregate_metrics, pred_df, gt_df).
    pred_df / gt_df are [N, C] DataFrames of binary class predictions.
    Reference labels come from a supplied dataframe or explicit CSV.
    Otherwise the reference reports are classified.
    """
    if not (len(preds) == len(gts) == len(image_ids)):
        raise ValueError(
            "Classifier evaluation inputs must have equal lengths: "
            f"preds={len(preds)}, gts={len(gts)}, image_ids={len(image_ids)}"
        )
    gt_df = get_reference_labels(
        gts,
        image_ids,
        LABEL_COLUMNS,
        predict_labels,
        default_gt_df=default_gt_df,
        gt_labels_path=gt_labels_path,
    )
    pred_df = predict_labels(preds)

    results = compute_label_metrics(
        pred_df,
        gt_df,
        label_columns=gt_df.columns,
        crg_label_columns=LABEL_COLUMNS,
        zero_division="warn",
    )
    return results, pred_df, gt_df
