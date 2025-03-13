import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from data.extractors import AMOSMMExtractor
from data.formatters import get_formatter

SSL_MODEL_NAME = "EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres"


class AMOSMMDataset(Dataset):
    """
    AMOS-MM dataset class
    Args:
        dataset_json_path (str): Path to the dataset JSON file
        model_name (str): Name of the vlm the dataset is used for. Used to infere the formatter for the questions
        split (str): Split name ('train' or 'val')
        embeddings_path (str | Path): Root path to the vision embeddings directory
    """

    def __init__(self, dataset_json_path, model_name, split="train", embeddings_path=None, n_samples=None):
        super(AMOSMMDataset).__init__()

        if embeddings_path is None:
            embeddings_path = os.environ.get("ROCKET_VLM_VISION_EMBEDDINGS")

        self.format_region_question = get_formatter(f"{model_name}_amos_regions")
        self.format_qa_data = get_formatter(f"{model_name}_amos_qa")

        self.split = split
        self.embeddings_path = embeddings_path

        with open(dataset_json_path, "r") as f:
            self.data = json.load(f)

        self.n_samples = n_samples
        self.samples = self.format_data()

    def format_data(self):
        if self.split == "train":
            samples = self.data["training"]
        elif self.split == "val":
            samples = self.data["validation"]
        else:
            raise ValueError(f"Invalid split name {self.split}")
        formatted_data = []

        if self.n_samples:
            samples = samples[: self.n_samples]

        for sample in samples:
            embedding_path = self.get_embedding_path(sample["image"])

            # Skip if embedding file doesn't exist
            if not os.path.exists(embedding_path):
                continue
                # raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

            # Use extractor to get all structured data
            extracted_data = AMOSMMExtractor.extract_sample(sample)

            item = {
                "image_path": extracted_data["image_path"],
                "embedding_path": embedding_path,
                "embedding": None,
                "report_text": [],
                "report_text_prompt_only": [],
                "qa_text": [],
                "qa_text_prompt_only": [],
                "item_qa": extracted_data["qa_items"],
                "item_report": extracted_data["report"] if extracted_data["report"] else {},
            }

            # Process report data using formatters
            if extracted_data["report"]:
                for region, region_data in extracted_data["report"]["region_questions"].items():
                    if region_data:
                        region_report, region_report_prompt_only = self.format_region_question(
                            region_data, embedding_path
                        )
                        if region_report:
                            item["report_text"].append(region_report)
                            item["report_text_prompt_only"].append(region_report_prompt_only)

            # Process QA data using formatters
            for qa_item in extracted_data["qa_items"]:
                formatted_qa, formatted_qa_prompt_only = self.format_qa_data(qa_item, embedding_path)
                item["qa_text"].append(formatted_qa)
                item["qa_text_prompt_only"].append(formatted_qa_prompt_only)

            # Skip samples with no report text or QA text
            if len(item["report_text"]) == 0 or len(item["qa_text"]) == 0:
                continue

            formatted_data.append(item)

        return formatted_data

    def get_embedding_path(self, image_path):
        """Convert image path to embedding path"""
        base_name = image_path.replace(".nii.gz", "")
        return str(self.embeddings_path / f"{base_name}_embedding.npy")

    def verify(self):
        """Test function to verify dataset loading"""
        print(f"\nTotal number of samples: {len(self)}")

        # Print example of region-specific question
        print("\nExample of region question:")
        for sample in self.samples:
            if "do you see" in sample[1]["content"][1]["text"]:
                print(f"Question: {sample[1]['content'][1]['text']}")
                print(f"Answer: {sample[2]['content'][0]['text']}")
                break

        # Print example of original QA
        print("\nExample of original QA:")
        for sample in self.samples:
            if "The answer is" in sample[2]["content"][0]["text"]:
                print(f"Question: {sample[1]['content'][1]['text']}")
                print(f"Answer: {sample[2]['content'][0]['text']}")
                break

    def __getitem__(self, index):
        sample = self.samples[index]

        # Create a new dictionary instead of modifying the stored one
        result = dict(sample)

        if sample["embedding_path"].endswith(".npy"):
            result["embedding"] = np.load(sample["embedding_path"])
        elif sample["embedding_path"].endswith(".npz"):
            npz_file = np.load(sample["embedding_path"])
            # Attempt to load the "embeddings" key if it exists, otherwise load the first array in the file
            if "embedding" in npz_file:
                result["embedding"] = npz_file["embeddings"]
            else:
                result["embedding"] = npz_file[npz_file.files[0]]
        else:
            raise ValueError(f"Unsupported embedding file format: {sample['embedding_path']}")
        return result

    def __len__(self):
        return len(self.samples)


class LIDCDataset(Dataset):
    """
    LIDC dataset class
    Args:
        dataset_json_path (str): Path where the dataset jsons are located.
            Should contain train_questions.json, val_questions.json, and test_questions.json
        model_name (str): Name of the vlm the dataset is used for. Used to infere the formatter for the questions
        split (str): Split name ('train', 'val', or 'test')
        embeddings_path (str | Path): Path to the vision embeddings directory
    """

    def __init__(
        self,
        dataset_json_path,
        model_name,
        split="train",
        embeddings_path=Path(os.getenv("ROCKET_VLM_VISION_EMBEDDINGS")),
    ):
        super(LIDCDataset).__init__()
        self.format_qa_data = get_formatter(f"{model_name}_lidc_qa")
        self.split = split
        if self.split == "train":
            dataset_json_path = os.path.join(dataset_json_path, "train_questions.json")
        elif self.split == "val":
            dataset_json_path = os.path.join(dataset_json_path, "val_questions.json")
        elif self.split == "test":
            dataset_json_path = os.path.join(dataset_json_path, "test_questions.json")
        else:
            raise ValueError(f"Invalid split name {self.split}")
        with open(dataset_json_path, "r") as f:
            self.data = json.load(f)
        self.embeddings_path = Path(embeddings_path) / "LIDC-IDRI" / "192_sliding_window" / SSL_MODEL_NAME / "npz"
        print(self.embeddings_path)
        print(self.embeddings_path.exists())
        self.samples = self.format_data()

    def get_embedding_path(self, series_uid):
        """Convert image path to embedding path"""
        base_name = series_uid.replace(".", "_")
        return str(Path(self.embeddings_path) / f"{base_name}_embedding.npz")

    def format_data(self):
        formatted_data = []
        for sample in self.data:
            embedding_path = self.get_embedding_path(sample["series_uid"])

            # Skip if embedding file doesn't exist
            if not os.path.exists(embedding_path):
                continue
                # raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

            # Format QA pairs
            formatted_qa = self.format_qa_data(embedding_path, sample)
            formatted_data.append(formatted_qa)
        return formatted_data

    def verify(self):
        """Test function to verify dataset loading"""
        print(f"\nTotal number of samples: {len(self)}")

        # Print example of region-specific question
        print("\nExample question:")
        print(self.samples[0])

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
