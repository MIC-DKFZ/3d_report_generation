import json
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from data.extractors import (
    AMOSMMExtractor,
    AMOSMMRegionExtractor,
    CTRATEExtractor,
    LIDCExtractor,
)
from utils import logger

SYSTEM_TEXT = "You are a medical imaging AI trained to analyze CT scans."


def format_into_message(
    question, answer, model_name, system_text=SYSTEM_TEXT, embedding_path=None
):

    system_message = {"role": "system", "content": [{"type": "text", "text": system_text}]}

    if model_name == "qwen":
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "embedding_path": embedding_path},
                {"type": "text", "text": question},
            ],
        }
    elif model_name == "owl":
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": f"<|image|>\n{question}"}],
        }
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    answer_message = {"role": "assistant", "content": [{"type": "text", "text": answer}]}

    full_conversation = [system_message, user_message, answer_message]
    prompt_only_conversation = [system_message, user_message]

    return full_conversation, prompt_only_conversation


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

    def __init__(
        self,
        dataset_json_path,
        model_name,
        split="train",
        embeddings_path=None,
        n_samples=None,
        tokenize_fn=None,
        answer_types=("CHOICE",),
        split_files=None,
        return_embedding=True,
        cache_file=None,
        embedding_constant=None,
        include_cls_text_gen=False,
    ):
        super().__init__()

        if embeddings_path is None:
            embeddings_path = os.environ.get("ROCKET_VLM_VISION_EMBEDDINGS")

        self.model_name = model_name
        self.split = split
        self.embeddings_path = (
            Path(embeddings_path) if isinstance(embeddings_path, str) else embeddings_path
        )
        self.n_samples = n_samples
        self.tokenize_fn = tokenize_fn
        self.split_files = split_files
        self.return_embedding = return_embedding
        self.embedding_constant = embedding_constant
        self.include_cls_text_gen = include_cls_text_gen
        self.cache_file = cache_file

        if isinstance(answer_types, str):
            answer_types = [
                answer_types
            ]  # Convert single string to list of strings for consistency

        self.answer_type_dict = {}
        for answer_type in answer_types:
            if isinstance(answer_type, str):
                self.answer_type_dict[answer_type] = None
            elif isinstance(answer_type, tuple):
                self.answer_type_dict[answer_type[0]] = answer_type[1]
            elif isinstance(answer_type, list):
                self.answer_type_dict[answer_type[0]] = answer_type[1:]
            elif isinstance(answer_type, dict):
                self.answer_type_dict.update(answer_type)

        if self.cache_file is not None and os.path.exists(self.cache_file):
            logger.info(f"Loading cached dataset from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                self.samples = pickle.load(f)
            logger.info(f"Loaded {len(self.samples)} samples from cache.")
            return

        # Load and process data
        self.data = self._load_data(dataset_json_path)

        self.samples = self.format_data()
        self.samples = self._filter_samples(self.samples)

        if self.n_samples:
            self.samples = self.samples[: self.n_samples]

        if self.tokenize_fn:
            self._tokenize_samples()

        if self.cache_file is not None and not os.path.exists(self.cache_file):
            logger.info(f"Caching dataset to {self.cache_file}")
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.samples, f)
            logger.info(f"Cached {len(self.samples)} samples.")

    @abstractmethod
    def _load_data(self, dataset_json_path):
        """Load data from JSON file(s)"""
        pass

    @abstractmethod
    def get_embedding_path(self, identifier):
        """Convert image identifier to embedding path"""
        pass

    @abstractmethod
    def extract_sample_list(self, samples):
        """Extract sample list from raw data"""
        pass

    def format_data(self):
        """Format raw data into structured samples"""
        formatted_data = []

        extracted_samples = self.extract_sample_list(self.data)

        for sample in tqdm(extracted_samples, desc="Formatting samples"):
            embedding_path = self.get_embedding_path(sample["image_id"])

            # Skip if embedding file doesn't exist
            if not os.path.exists(embedding_path):
                continue

            sample["embedding_path"] = embedding_path

            formatted_qa, formatted_qa_prompt_only = format_into_message(
                sample["question"],
                sample["full_answer"],
                self.model_name,
                system_text=sample["system_text"],
            )

            sample["formatted_qa"] = formatted_qa
            sample["formatted_qa_prompt_only"] = formatted_qa_prompt_only

            formatted_data.append(sample)

        return formatted_data

    def _tokenize_samples(self):
        for sample in tqdm(self.samples, desc="Tokenizing samples"):
            # sample["tokenized"] = {}
            sample["tokenized_qa_text_tokens"] = self.tokenize_fn(sample["formatted_qa"])
            sample["tokenized_qa_text_prompt_only_tokens"] = self.tokenize_fn(
                sample["formatted_qa_prompt_only"]
            )

    def _filter_samples(self, samples):
        """Filter samples based on answer type"""
        filtered_samples = []

        if self.answer_type_dict is None:
            return samples  # If answer_type_dict is empty, return all samples
        if len(self.answer_type_dict) == 0:
            return samples  # If answer_type_dict is not empty, filter samples

        for sample in samples:
            if sample["answer_type"] in self.answer_type_dict:
                if self.answer_type_dict[sample["answer_type"]] is not None:
                    if sample["answer_subtype"] in self.answer_type_dict[sample["answer_type"]]:
                        filtered_samples.append(sample)
                else:
                    filtered_samples.append(sample)

        return filtered_samples

    def _load_embedding(self, embedding_path):
        """Load embedding from file"""
        try:
            if embedding_path.endswith(".npy"):
                return np.load(embedding_path)
            elif embedding_path.endswith(".npz"):
                # Use context manager to ensure proper file closure
                with np.load(embedding_path) as npz_file:
                    # Attempt to load the "embeddings" key if it exists, otherwise load the first array
                    if "embedding" in npz_file:
                        embedding = npz_file[
                            "embeddings"
                        ].copy()  # Make a copy to avoid reference issues
                    else:
                        embedding = npz_file[
                            npz_file.files[0]
                        ].copy()  # Make a copy to avoid reference issues
                return embedding
            else:
                raise ValueError(f"Unsupported embedding file format: {embedding_path}")
        except Exception as e:
            logger.error(f"Error loading embedding from {embedding_path}: {e}")
            return None

    def _embedding_2_tensor(self, embedding):
        """Convert embedding to tensor"""

        if embedding is None:
            logger.error("Embedding is None, cannot convert to tensor.")
            return None

        np_img_flat = embedding.reshape(embedding.shape[0], embedding.shape[1], -1)
        np_img_flat = np_img_flat.transpose(0, 2, 1)
        img_tensor = torch.tensor(np_img_flat)

        return img_tensor

    def __getitem__(self, index):
        sample = self.samples[index]

        # Create a new dictionary instead of modifying the stored one
        result = dict(sample)

        if self.return_embedding:
            # Load embedding and convert to tensor
            if self.embedding_constant is None:
                if "embedding_path" not in sample:
                    raise ValueError(f"Sample {index} does not have an embedding path.")
                result["embedding"] = self._load_embedding(sample["embedding_path"])
            else:
                result["embedding"] = np.full((1, 864, 24, 24, 24), self.embedding_constant)
            result["embedding_tensor"] = self._embedding_2_tensor(result["embedding"])
            if result["embedding_tensor"] is not None:
                result["embedding_tensor"] = result["embedding_tensor"].half()  # Ensure tensor is
            if result["embedding"] is not None:
                result["embedding"] = result["embedding"].astype(
                    np.float16
                )  # Convert to float16 for memory efficiency

        return result

    def __len__(self):
        return len(self.samples)

    def verify(self):
        """Test function to verify dataset loading"""
        logger.info(f"\nTotal number of samples: {len(self)}")
        logger.info("\nExample question:")
        if len(self.samples) > 0:
            sample = self.samples[0]
            logger.info(f"Image ID: {sample['image_id']}")
            logger.info(f"Question: {sample['question']}")
            logger.info(f"Answer: {sample['full_answer']}")
            logger.info(f"Answer Type: {sample['answer_type']}")
            logger.info(f"Answer Subtype: {sample['answer_subtype']}")


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
            data = json.load(f)

        if self.split == "train":
            samples = data["training"]
        elif self.split == "val":
            samples = data["validation"]
        else:
            raise ValueError(f"Invalid split name {self.split}")

        return samples

    def get_embedding_path(self, image_path):
        """Convert image path to embedding path"""
        base_name = image_path.replace(".nii.gz", "")
        return str(os.path.join(self.embeddings_path, f"{base_name}_embedding.npy"))

    def extract_sample_list(self, samples):
        """Extract sample list from raw data"""
        extracted_samples = AMOSMMExtractor.extract_sample_list(samples)
        return extracted_samples


class AMOSMMRegionDataset(AMOSMMDataset):
    """
    AMOS-MM Region based dataset class

    Args:
        dataset_json_path (str): Path to the dataset JSON file
        model_name (str): Name of the VLM the dataset is used for
        split (str): Split name ('train' or 'val')
        embeddings_path (str | Path): Root path to the vision embeddings directory
        n_samples (int, optional): Number of samples to load (for debugging)
    """

    def format_data(self):
        """Format raw data into structured samples"""
        formatted_data = []

        extracted_samples = self.extract_sample_list(self.data)

        for sample in extracted_samples:
            embedding_path = self.get_embedding_path(sample["image_id"], sample["region"])

            # Skip if embedding file doesn't exist
            if not os.path.exists(embedding_path):
                continue

            sample["embedding_path"] = embedding_path

            formatted_qa, formatted_qa_prompt_only = format_into_message(
                sample["question"],
                sample["full_answer"],
                self.model_name,
                system_text=sample["system_text"],
            )

            sample["formatted_qa"] = formatted_qa
            sample["formatted_qa_prompt_only"] = formatted_qa_prompt_only

            formatted_data.append(sample)

        return formatted_data

    def get_embedding_path(self, image_path, region):
        """Convert image path to embedding path"""
        base_name = image_path.replace(".nii.gz", "")
        return str(os.path.join(self.embeddings_path, f"{base_name}_{region}_embedding.npz"))

    def extract_sample_list(self, samples):
        """Extract sample list from raw data"""
        # TODO: Currently this class could be implemented in the ABC class as well.
        # The extractor class is then defined in the init of the class
        extracted_samples = AMOSMMRegionExtractor.extract_sample_list(samples)
        return extracted_samples


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

    def get_embedding_path(self, series_uid):
        """Convert image path to embedding path"""
        base_name = series_uid.replace(".", "_")
        return str(Path(self.embeddings_path) / f"{base_name}_embedding.npz")

    def extract_sample_list(self, samples):
        """Extract sample list from raw data"""
        extracted_samples = LIDCExtractor.extract_sample_list(samples)
        return extracted_samples


class CTRATEDataset(MedicalImageDataset):
    """
    CT-RATE dataset class for CT scan analysis

    Args:
        dataset_json_path (str): Path to the CT-RATE dataset directory
        model_name (str): Name of the VLM the dataset is used for
        split (str): Split name ('train', 'val', or 'test')
        embeddings_path (str | Path): Path to the vision embeddings directory
        n_samples (int, optional): Number of samples to load (for debugging)
        answer_types (tuple, optional): Types of answers to include
    """

    # def __init__(
    #     self,
    #     dataset_path,
    #     model_name,
    #     split="train",
    #     embeddings_path=None,
    #     n_samples=None,
    #     tokenize_fn=None,
    #     answer_types=("CHOICE", "TEXT"),
    # ):
    #     super().__init__(
    #         dataset_path,
    #         model_name,
    #         split=split,
    #         embeddings_path=embeddings_path,
    #         n_samples=n_samples,
    #         tokenize_fn=tokenize_fn,
    #         answer_types=answer_types,
    #     )

    def _load_data(self, dataset_path):

        self.embedding_dict = {}

        for flr_, dirs_, fls_ in os.walk(self.embeddings_path):
            for fl in fls_:
                if fl.endswith(".npy") or fl.endswith(".npz"):
                    image_id = fl.split("_embedding")[0]
                    self.embedding_dict[image_id] = os.path.join(flr_, fl)

        metadata_dir = os.path.join(dataset_path, "metadata")
        data_dir = os.path.join(dataset_path, "images")

        metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith(".yml")]

        x_val_split_data = None
        if self.split_files:
            if os.path.exists(self.split_files):
                with open(self.split_files, "r") as f:
                    x_val_split_data = json.load(f)

        if x_val_split_data is not None and "_" in self.split:
            fold_num = int(self.split.split("_")[1])
            train_ids = set(x_val_split_data[f"fold_{fold_num}"]["train_ids"])
            val_ids = set(x_val_split_data[f"fold_{fold_num}"]["val_ids"])

            if self.split.startswith("val"):
                # If the split is a validation split, filter metadata files to only include validation IDs
                metadata_files = [f for f in metadata_files if f.replace(".yml", "") in val_ids]
            else:
                metadata_files = [f for f in metadata_files if f.replace(".yml", "") in train_ids]

        samples = []
        for metadata_file in tqdm(metadata_files, desc="Loading metadata files"):
            metadata_path = os.path.join(metadata_dir, metadata_file)
            # data_path = os.path.join(data_dir, metadata_file.replace(".yml", ".nrrd"))

            # if not os.path.exists(data_path):
            #     continue

            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)

            image_id = metadata["image_id"]

            embedding_path = self.embedding_dict.get(image_id)

            if embedding_path is None:
                continue

            if self.split == "train" and not metadata["metadata"].get("train_set", False):
                continue
            elif self.split == "val" and metadata["metadata"].get("train_set", True):
                continue

            findings = metadata["metadata"]["Findings_EN"]
            impressions = metadata["metadata"]["Impressions_EN"]
            classifications = {c: metadata["metadata"].get(c) for c in CTRATEExtractor.CONDITIONS}

            sample = {
                "image_id": metadata["image_id"],
                "embedding_path": embedding_path,
                "tasks": {
                    "classification": classifications,
                    "report_gen": findings,
                    "diagnosis": impressions,
                },
            }
            samples.append(sample)

        return samples

    def get_embedding_path(self, embedding_path):
        """Convert image ID to embedding path"""
        if not hasattr(self, "embedding_dict"):
            logger.error("embedding_dict not found. Ensure _load_data is called first.")
            return os.path.join(str(self.embeddings_path), embedding_path + "_embedding.npz")

        return self.embedding_dict.get(embedding_path, None)
        # return os.path.join(self.embeddings_path, embedding_path + "_embedding.npz")

    def extract_sample_list(self, samples):
        """Extract sample list from raw data"""
        extracted_samples = CTRATEExtractor.extract_sample_list(
            samples, include_cls_text_gen=self.include_cls_text_gen
        )
        return extracted_samples
