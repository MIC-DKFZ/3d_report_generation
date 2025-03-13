import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from data.extractors import AMOSMMExtractor
from data.formatters import get_formatter


class AMOSMMDataset(Dataset):
    """
    AMOS-MM dataset class
    Args:
        dataset_json_path (str): Path to the dataset JSON file
        model_name (str): Name of the vlm the dataset is used for. Used to infere the formatter for the questions
        split (str): Split name ('train' or 'val')
        embeddings_path (str): Path to the vision embeddings directory
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
        return str(Path(self.embeddings_path) / f"{base_name}_embedding.npy")

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
