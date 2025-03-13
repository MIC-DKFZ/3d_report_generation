import json
import os
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset
from data.extractors import AMOSMMExtractor, LIDCExtractor
from data.formatters import get_formatter


class MedicalImageDataset(Dataset, ABC):
    """
    Base class for medical imaging datasets with vision-language models

    Args:
        dataset_json_path (str): Path to the dataset JSON file or directory
        model_name (str): Name of the VLM the dataset is used for
        split (str): Split name ('train', 'val' or 'test')
        embeddings_path (str | Path): Root path to the vision embeddings directory
        n_samples (int, optional): Number of samples to load (for debugging)
    """

    def __init__(self, dataset_json_path, model_name, split="train", embeddings_path=None, n_samples=None):
        super().__init__()

        if embeddings_path is None:
            embeddings_path = os.environ.get("ROCKET_VLM_VISION_EMBEDDINGS")

        self.model_name = model_name
        self.split = split
        self.embeddings_path = Path(embeddings_path) if isinstance(embeddings_path, str) else embeddings_path
        self.n_samples = n_samples

        # Load and process data
        self.data = self._load_data(dataset_json_path)
        self._init_formatters()
        self.samples = self.format_data()

    @abstractmethod
    def _load_data(self, dataset_json_path):
        """Load data from JSON file(s)"""
        pass

    @abstractmethod
    def _init_formatters(self):
        """Initialize formatters based on model name"""
        pass

    @abstractmethod
    def format_data(self):
        """Format raw data into structured samples"""
        pass

    @abstractmethod
    def get_embedding_path(self, identifier):
        """Convert image identifier to embedding path"""
        pass

    @abstractmethod
    def verify(self):
        """Test function to verify dataset loading"""
        pass

    def _load_embedding(self, embedding_path):
        """Load embedding from file"""
        if embedding_path.endswith(".npy"):
            return np.load(embedding_path)
        elif embedding_path.endswith(".npz"):
            npz_file = np.load(embedding_path)
            # Attempt to load the "embeddings" key if it exists, otherwise load the first array
            if "embedding" in npz_file:
                return npz_file["embeddings"]
            else:
                return npz_file[npz_file.files[0]]
        else:
            raise ValueError(f"Unsupported embedding file format: {embedding_path}")

    def __getitem__(self, index):
        sample = self.samples[index]

        # Create a new dictionary instead of modifying the stored one
        result = dict(sample)
        result["embedding"] = self._load_embedding(sample["embedding_path"])

        return result

    def __len__(self):
        return len(self.samples)


class AMOSMMDataset(MedicalImageDataset):
    """
    AMOS-MM dataset class

    Args:
        dataset_json_path (str): Path to the dataset JSON file
        model_name (str): Name of the VLM the dataset is used for
        split (str): Split name ('train' or 'val')
        embeddings_path (str | Path): Root path to the vision embeddings directory
        n_samples (int, optional): Number of samples to load (for debugging)
    """

    def _load_data(self, dataset_json_path):
        json_file_path = os.path.join(dataset_json_path, "dataset.json")
        with open(json_file_path, "r") as f:
            return json.load(f)

    def _init_formatters(self):
        self.format_region_question = get_formatter(f"{self.model_name}_amos_regions")
        self.format_qa_data = get_formatter(f"{self.model_name}_amos_qa")

    def get_embedding_path(self, image_path):
        """Convert image path to embedding path"""
        base_name = image_path.replace(".nii.gz", "")
        return str(os.path.join(self.embeddings_path, f"{base_name}_embedding.npy"))

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

    def verify(self):
        """Test function to verify dataset loading"""
        print(f"\nTotal number of samples: {len(self)}")

        # Print example of region-specific question
        print("\nExample of region question:")
        for sample in self.samples:
            if "do you see" in sample["qa_text"][0]:
                print(f"Question: {sample['qa_text'][0]}")
                break

        # Print example of original QA
        print("\nExample of original QA:")
        for sample in self.samples:
            if "The answer is" in sample["qa_text"][0]:
                print(f"Question: {sample['qa_text'][0]}")
                break


class LIDCDataset(MedicalImageDataset):
    """
    LIDC dataset class

    Args:
        dataset_json_path (str): Path where the dataset jsons are located
        model_name (str): Name of the VLM the dataset is used for
        split (str): Split name ('train', 'val', or 'test')
        embeddings_path (str | Path): Path to the vision embeddings directory
        n_samples (int, optional): Number of samples to load (for debugging)
    """

    def _load_data(self, dataset_json_path):
        if self.split == "train":
            file_path = os.path.join(dataset_json_path, "train_questions.json")
        elif self.split == "val":
            file_path = os.path.join(dataset_json_path, "val_questions.json")
        elif self.split == "test":
            file_path = os.path.join(dataset_json_path, "test_questions.json")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        with open(file_path, "r") as f:
            return json.load(f)

    def _init_formatters(self):
        self.format_qa_data = get_formatter(f"{self.model_name}_lidc_qa")

    def get_embedding_path(self, series_uid):
        """Convert image path to embedding path"""
        base_name = series_uid.replace(".", "_")
        return str(Path(self.embeddings_path) / f"{base_name}_embedding.npz")

    def format_data(self):
        formatted_data = []
        samples = self.data

        if self.n_samples:
            samples = samples[: self.n_samples]

        for sample in samples:
            embedding_path = self.get_embedding_path(sample["series_uid"])

            # Skip if embedding file doesn't exist
            if not os.path.exists(embedding_path):
                continue

            # Extract data using the LIDCExtractor
            extracted_data = LIDCExtractor.extract_sample(sample)

            # Format QA pairs
            formatted_qa, formatted_qa_prompt_only = self.format_qa_data(extracted_data["item_qa"], embedding_path)

            item = {
                "series_uid": extracted_data["series_uid"],
                "subject_id": extracted_data["subject_id"],
                "scan_id": extracted_data["scan_id"],
                "study_uid": extracted_data["study_uid"],
                "embedding_path": embedding_path,
                "embedding": None,
                "qa_text": [formatted_qa],
                "qa_text_prompt_only": [formatted_qa_prompt_only],
                "item_qa": extracted_data["item_qa"],
            }

            formatted_data.append(item)

        return formatted_data

    def verify(self):
        """Test function to verify dataset loading"""
        print(f"\nTotal number of samples: {len(self)}")
        print("\nExample question:")
        if len(self.samples) > 0:
            sample = self.samples[0]
            print(f"Series UID: {sample['series_uid']}")
            print(f"Question: {sample['qa_text'][0]}")
