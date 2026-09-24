"""Merlin ModernBERT report classifier — RadBERT-equivalent for the Merlin dataset.

Mirrors ``vlm.evaluation.ctrate_radbert.utils`` but targets a fine-tuned
ModernBERT multi-label classifier (trained in the MerlinBert project). Given a
list of report strings it returns a [N, 30] matrix of binary present/absent
predictions, one column per finding in ``merlin_label_cols`` order.

Unlike RadBERT (a custom nn.Module + ``.pth`` state dict that reads
``pooler_output``), ModernBERT has no pooler, so the model is a standard
``AutoModelForSequenceClassification`` saved with ``save_pretrained``. Point
``VLM_MERLIN_MODERNBERT_DIR`` at that self-contained directory.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from torch.utils.data import Dataset

merlin_label_cols = [
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

    def __init__(self, n_classes=len(merlin_label_cols), hf_model_dir=None):
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


def load_label_cols(model_dir):
    """Return the label order saved alongside the model (falls back to module default)."""
    f = Path(model_dir) / "label_cols.json"
    if f.exists():
        return json.loads(f.read_text())
    return merlin_label_cols
