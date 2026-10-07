"""Evaluate generated CT-RATE reports using RadBERT finding labels."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoConfig, AutoModel, AutoTokenizer

from vlm.evaluation.report_generation.utils import (
    compute_label_metrics,
    load_reference_labels,
)
from vlm.utils import logger


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
    def __init__(self, data_files, class_count, label_cols, hf_model_dir=None):
        self.data_files = data_files
        self.class_count = class_count

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, do_lower_case=True)
        self.max_length = 512
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
    def __init__(self, n_classes=10, hf_model_dir=None):
        super().__init__()

        self.config = AutoConfig.from_pretrained(hf_model_dir)
        self.model = AutoModel.from_pretrained(hf_model_dir, config=self.config)

        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attn_mask):
        # ensure input_ids and attention_mask are 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)
        output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)

        return output


class RadBertInference:
    def __init__(
        self,
        model,
        dataloaders,
        num_class,
        test_label_cols,
        threshold=0.5,
    ):

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
                input_ids = torch.squeeze(input_ids)
                attention_mask = torch.squeeze(attention_mask)
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")

                logits = self.model(input_ids, attention_mask)

                pred_logits = np.concatenate(
                    (pred_logits, logits.detach().cpu().numpy()), axis=0
                )

        pred_logits = pred_logits[1:]
        pred_labels = expit(pred_logits)

        pred_labels[pred_labels >= self.threshold] = 1
        pred_labels[pred_labels < self.threshold] = 0

        return pred_labels


def predict_labels(reports: list[str], ckpt_path: str | Path, hf_radbert_dir: str | Path):
    df = pd.DataFrame({"reports": reports})
    for col in LABEL_COLUMNS:
        df[col] = 0

    batch_size = 32
    if os.getenv("SLURM_JOB_ID") or os.getenv("LSB_JOBID"):
        batch_size = 96

    # DataLoader
    ds = CTDataset(df, len(LABEL_COLUMNS), LABEL_COLUMNS, hf_radbert_dir)
    loader = DataLoader(
        ds, sampler=SequentialSampler(ds), batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # Model
    model = RadBertClassifier(n_classes=len(LABEL_COLUMNS), hf_model_dir=hf_radbert_dir)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model.embeddings.position_ids" in state_dict:
        del state_dict["model.embeddings.position_ids"]

    model.load_state_dict(state_dict)
    model = model.cuda().eval()

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
) -> tuple[dict, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Returns (aggregate_metrics, pred_df, gt_df).
    pred_df / gt_df are [N, C] DataFrames of binary class predictions; None when RadBERT is unavailable.
    """
    if not (len(preds) == len(gts) == len(image_ids)):
        raise ValueError(
            "Classifier evaluation inputs must have equal lengths: "
            f"preds={len(preds)}, gts={len(gts)}, image_ids={len(image_ids)}"
        )
    radbert_dir = os.getenv("VLM_CTRATE_RADBERT_DIR")
    if radbert_dir is None:
        logger.warning_main(
            "The environment variable 'VLM_CTRATE_RADBERT_DIR' is not set. "
            "Cannot proceed with RadBERT inference for CT-RATE."
        )
        return {}, None, None

    radbert_dir = Path(radbert_dir)
    radbert_ckpt = radbert_dir / "RadBertClassifier.pth"
    hf_radbert_dir = radbert_dir / "RadBERT-RoBERTa-4m"

    pred_df = predict_labels(preds, radbert_ckpt, hf_radbert_dir)
    gt_labels_path = radbert_dir / "gt_labels.csv"
    if default_gt_df is not None:
        gt_df = default_gt_df.reset_index(drop=True)
    elif gt_labels_path.is_file():
        logger.info_main(f"Loading CT-RATE ground-truth labels from {gt_labels_path}")
        gt_df = load_reference_labels(gt_labels_path, image_ids, LABEL_COLUMNS)
    else:
        gt_df = predict_labels(gts, radbert_ckpt, hf_radbert_dir)

    results = compute_label_metrics(
        pred_df,
        gt_df,
        label_columns=gt_df.columns,
        crg_label_columns=LABEL_COLUMNS,
        zero_division="warn",
    )
    return results, pred_df, gt_df
