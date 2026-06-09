from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch.nn as nn
import torch
import numpy as np
from scipy.special import expit

ctrate_label_cols = [
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

    def __init__(
            self,
            data_files,
            class_count,
            label_cols,
            hf_model_dir=None
    ):
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