import json
import os
import pickle
import random
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import yaml
import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import nrrd
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from data.extractors import (
    AMOSMMExtractor,
    AMOSMMRegionExtractor,
    CTRATEExtractor,
    LIDCExtractor,
    QAItemDict,
)
from utils import logger
from typing import Type
from numbers import Number

SYSTEM_TEXT = "You are a medical imaging AI trained to analyze CT scans."


def convert_dtype(image: np.ndarray, target_dtype: Type[Number]) -> np.ndarray:
    return image.astype(target_dtype)


def z_score_normalization(
    image: np.ndarray, use_mask_for_norm: bool, non_zero_mask: np.ndarray, target_dtype: Type[Number]
) -> np.ndarray:
    image = convert_dtype(image, target_dtype)
    if use_mask_for_norm:
        mask = non_zero_mask >= 0
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / max(std, 1e-8)
    else:
        mean = image.mean()
        std = image.std()
        image = (image - mean) / max(std, 1e-8)
    return image


def format_into_message(
    question, answer, model_name, system_text=SYSTEM_TEXT, embedding_path=None
):

    system_message = {"role": "system", "content": [{"type": "text", "text": system_text}]}

    if model_name == "qwen":
        user_message = {
            "role": "user",
            "content": [
                {"type": "video", "embedding_path": embedding_path},
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


def _process_sample_for_formatting(sample, get_embedding_path_func):
    """Helper function to process a single sample for formatting"""
    embedding_path = get_embedding_path_func(sample["image_id"])
    
    # Skip if embedding file doesn't exist
    if not os.path.exists(embedding_path):
        return None
    
    sample["embedding_path"] = embedding_path
    return sample


def _tokenize_single_sample(sample_data, tokenize_fn):
    """Helper function to tokenize a single sample"""
    sample, formatted_qa, formatted_qa_prompt_only = sample_data
    sample["formatted_qa"] = formatted_qa
    sample["formatted_qa_prompt_only"] = formatted_qa_prompt_only
    sample["tokenized_qa_text_tokens"] = tokenize_fn(formatted_qa)
    sample["tokenized_qa_text_prompt_only_tokens"] = tokenize_fn(formatted_qa_prompt_only)
    return sample


def _load_yaml_file(file_info):
    """Helper function to load a single YAML file for CTRATEDataset"""
    metadata_file, metadata_dir, data_dir, embedding_dict, split, vqa_path = file_info
    
    metadata_path = os.path.join(metadata_dir, metadata_file)
    
    try:
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading {metadata_path}: {e}")
        return None

    image_id = metadata["image_id"]
    embedding_path = embedding_dict.get(image_id)

    if embedding_path is None:
        return None

    if split == "train" and not metadata["metadata"].get("train_set", False):
        return None
    elif split == "val" and metadata["metadata"].get("train_set", True):
        return None

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
    
    # Load VQA data if vqa_path is provided
    if vqa_path is not None:
        try:
            # Extract case identifier from image filepath
            image_filepaths = metadata.get("image_filepaths", [])
            if image_filepaths:
                # Get the filename from the first filepath
                # Example: .../train_14008_a_2.nii.gz -> train_14008_a_2.nii.gz
                filepath = image_filepaths[0]
                filename = os.path.basename(filepath)
                
                # Remove file extension and trailing number+underscore
                # train_14008_a_2.nii.gz -> train_14008_a_2 -> train_14008_a
                case_dir = filename.split('.')[0]  # Remove .nii.gz
                case_dir = '_'.join(case_dir.split('_')[:-1])  # Remove _2 (last part)
                
                # Determine VQA folder based on case type
                if case_dir.startswith('train_'):
                    vqa_folder = "train_cases"
                elif case_dir.startswith('valid_'):
                    vqa_folder = "validation_cases"
                else:
                    vqa_folder = None
                
                if vqa_folder:
                    vqa_file_path = os.path.join(vqa_path, vqa_folder, f"{case_dir}.json")
                    
                    if os.path.exists(vqa_file_path):
                        with open(vqa_file_path, "r") as f:
                            vqa_data = json.load(f)
                        sample["vqa_data"] = vqa_data.get("qa_pairs", [])
                    else:
                        print(f"VQA file not found: {vqa_file_path}")
                        logger.debug(f"VQA file not found: {vqa_file_path}")
                            
        except Exception as e:
            logger.error(f"Error loading VQA data for {image_id}: {e}")
    
    return sample


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
        cls_aug_strength=0.0,
        load_image=False,
        max_workers=None,
        vqa_path=None,
    ):
        super().__init__()

        if embeddings_path is None:
            embeddings_path = os.environ.get("ROCKET_VLM_VISION_EMBEDDINGS")

        self.model_name = model_name
        self.split = split
        self.embeddings_path = (
            Path(embeddings_path) if isinstance(embeddings_path, str) else embeddings_path
        )
        self.vqa_path = vqa_path
        self.n_samples = n_samples
        self.tokenize_fn = tokenize_fn
        self.split_files = split_files
        self.return_embedding = return_embedding
        self.embedding_constant = embedding_constant
        self.include_cls_text_gen = include_cls_text_gen
        self.cls_aug_strength = cls_aug_strength
        self.cache_file = cache_file
        self.load_image = load_image
        self.max_workers = max_workers or min(48, (os.cpu_count() or 1))
        logger.info(f"Using {self.max_workers} for dataset loading.")
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
        """Format raw data into structured samples with parallel processing"""
        extracted_samples = self.extract_sample_list(self.data)
        
        logger.info(f"Processing {len(extracted_samples)} samples with parallel formatting...")
        
        # Use ThreadPoolExecutor for I/O bound operations (file existence checks)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            process_func = partial(_process_sample_for_formatting, get_embedding_path_func=self.get_embedding_path)
            future_to_sample = {executor.submit(process_func, sample): sample for sample in extracted_samples}
            
            formatted_data = []
            for future in tqdm(as_completed(future_to_sample), total=len(extracted_samples), desc="Formatting samples"):
                result = future.result()
                if result is not None:
                    formatted_data.append(result)

        logger.info(f"Formatted {len(formatted_data)} samples successfully")
        return formatted_data

    def _tokenize_samples(self):
        """Tokenize samples with parallel processing"""
        logger.info("Preparing samples for tokenization...")
        
        # Prepare data for tokenization
        sample_data = []
        for sample in self.samples:
            formatted_qa, formatted_qa_prompt_only = format_into_message(
                sample["question"],
                sample["full_answer"],
                self.model_name,
                system_text=sample["system_text"],
            )
            sample_data.append((sample, formatted_qa, formatted_qa_prompt_only))
        
        logger.info(f"Tokenizing {len(sample_data)} samples in parallel...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tokenize_func = partial(_tokenize_single_sample, tokenize_fn=self.tokenize_fn)
            future_to_idx = {executor.submit(tokenize_func, data): idx for idx, data in enumerate(sample_data)}
            
            for future in tqdm(as_completed(future_to_idx), total=len(sample_data), desc="Tokenizing samples"):
                idx = future_to_idx[future]
                self.samples[idx] = future.result()

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

    def _load_embedding(self, embedding_path, is_image=True):
        """Load embedding from file"""
        try:
            if embedding_path.endswith(".npy"):
                embedding = np.load(embedding_path)
            elif embedding_path.endswith(".npz"):
                # Use context manager to ensure proper file closure
                with np.load(embedding_path) as npz_file:
                    # Attempt to load the "embeddings" key if it exists, otherwise load the first array
                    if "embedding" in npz_file:
                        embedding = npz_file["embeddings"].copy()  # Make a copy to avoid reference issues
                    else:
                        embedding = npz_file[npz_file.files[0]].copy()  # Make a copy to avoid reference issues

            elif embedding_path.endswith(".b2nd"):
                # Load b2nd file like nnUNet :)
                dparams = {
                    'nthreads': 1
                }
                mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}
                
                embedding = np.array(blosc2.open(urlpath=embedding_path, mode='r', dparams=dparams, **mmap_kwargs))
                
            elif embedding_path.endswith(".nrrd"):     
                embedding, header = nrrd.read(embedding_path)

            else:
                raise ValueError(f"Unsupported embedding file format: {embedding_path}")
            
            if self.load_image:
                embedding = z_score_normalization(embedding, False, None, np.float32)
                embedding = self._preprocess_3d_image(embedding)

            return embedding
            
        except Exception as e:
            logger.error(f"Error loading embedding from {embedding_path}: {e}")
            return None


    def _preprocess_3d_image(self, image_array):
        """Preprocess 3D image: convert to float32, remove singleton dims, pad if needed, and random crop to final patch size"""
        # Ensure float32 dtype
        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array, dtype=np.float32)
        else:
            image_array = image_array if image_array.dtype == np.float32 else image_array.astype(np.float32)
        
        # Remove singleton first dimension if present
        if image_array.ndim == 4 and image_array.shape[0] == 1:
            image_array = image_array.squeeze(0)
        
        # Final patch size
        patch_size = (192, 192, 192)
        
        # Get image dimensions (assuming format: H, W, D)
        h, w, d = image_array.shape
        
        # Pad if image is smaller than patch_size
        pad_h = max(0, patch_size[0] - h)
        pad_w = max(0, patch_size[1] - w) 
        pad_d = max(0, patch_size[2] - d)
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            # Split padding evenly on both sides to center the image
            pad_h_before = pad_h // 2
            pad_h_after = pad_h - pad_h_before
            
            pad_w_before = pad_w // 2
            pad_w_after = pad_w - pad_w_before
            
            pad_d_before = pad_d // 2
            pad_d_after = pad_d - pad_d_before
            
            # Pad the image
            padding = ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (pad_d_before, pad_d_after))
            image_array = np.pad(image_array, padding, mode='constant', constant_values=0)
        
        # Random crop to exact patch size (if larger than patch_size)
        h, w, d = image_array.shape
        if h > patch_size[0] or w > patch_size[1] or d > patch_size[2]:
            # Random patch extraction (same logic as was in processor)
            max_starts = [max(0, image_array.shape[i] - patch_size[i]) for i in range(3)]
            start_h, start_w, start_d = [int(np.clip(np.random.normal(m/2, m*0.1), 0, m)) for m in max_starts]
            
            # Extract random patch
            image_array = image_array[
                start_h:start_h + patch_size[0],
                start_w:start_w + patch_size[1], 
                start_d:start_d + patch_size[2]
            ]
        
        return image_array

    def _embedding_2_tensor(self, embedding):
        """Convert embedding to tensor"""
        # TODO check if we can remove this function!
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

        if self.include_cls_text_gen and "cls_text_gen" in sample:
            for key in result["cls_text_gen"]:
                if random.random() < self.cls_aug_strength:
                    result["cls_text_gen"][key] = 1 - result["cls_text_gen"][key]  # flip 0 to 1
            result["question"] += f' The scan is classified into the following classes {str(result["cls_text_gen"])}, where 1 means presence of a disease and 0 means absence.'
            # result["cls_text_gen"] = sample["cls_text_gen"]

        formatted_qa, formatted_qa_prompt_only = format_into_message(
                result["question"],
                result["full_answer"],
                self.model_name,
                system_text=sample["system_text"],
            )

        result["formatted_qa"] = formatted_qa
        result["formatted_qa_prompt_only"] = formatted_qa_prompt_only

        if self.return_embedding:
            # Load embedding and convert to tensor
            if self.embedding_constant is None:
                if "embedding_path" not in sample:
                    raise ValueError(f"Sample {index} does not have an embedding path.")
                result["embedding"] = self._load_embedding(sample["embedding_path"])
            else:
                if type(self.embedding_constant) is float:
                    result["embedding"] = np.full((self._load_embedding(sample["embedding_path"])).shape, self.embedding_constant)
                elif type(self.embedding_constant) is str and self.embedding_constant == "random":
                    result["embedding"] = np.random.rand(*(self._load_embedding(sample["embedding_path"])).shape)
                else:
                    raise ValueError(f"Unsupported embedding constant type: {type(self.embedding_constant)}")
                
            if self.load_image:
                result["embedding_tensor"] = result["embedding"]
            else:
                # do we really need this function? why do reshaping and why use torch.tensor()?
                result["embedding_tensor"] = self._embedding_2_tensor(result["embedding"])

            # TODO check if we can remove this stuff!
            #if result["embedding_tensor"] is not None:
            #    result["embedding_tensor"] = result["embedding_tensor"].half()  # Ensure tensor is
            #if result["embedding"] is not None:
            #    result["embedding"] = result["embedding"].astype(
            #        np.float16
            #    )  # Convert to float16 for memory efficiency

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
        vqa_path (str, optional): Path to the VQA data directory
    """

    def _load_data(self, dataset_path):
        logger.info("Building embedding dictionary...")
        
        # Build embedding dictionary with parallel processing
        self.embedding_dict = {}
        
        # Collect all embedding files first
        embedding_files = []
        for flr_, dirs_, fls_ in os.walk(self.embeddings_path):
            for fl in fls_:
                if fl.endswith((".npy", ".npz", ".b2nd", ".nrrd")):
                    embedding_files.append(os.path.join(flr_, fl))
        
        logger.info(f"Found {len(embedding_files)} embedding files")
        
        # Process embedding files to build dictionary
        for embedding_path in tqdm(embedding_files, desc="Building embedding dictionary"):
            fl = os.path.basename(embedding_path)
            image_id = fl.split("_embedding")[0].split(".")[0]
            self.embedding_dict[image_id] = embedding_path

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

        logger.info(f"Processing {len(metadata_files)} metadata files in parallel...")
        
        file_info_list = [(metadata_file, metadata_dir, data_dir, self.embedding_dict, self.split, self.vqa_path) 
                         for metadata_file in metadata_files]
        
        # Use ThreadPoolExecutor for I/O bound YAML reading
        samples = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(_load_yaml_file, file_info): file_info[0] 
                             for file_info in file_info_list}
            
            for future in tqdm(as_completed(future_to_file), total=len(file_info_list), desc="Loading metadata files"):
                result = future.result()
                if result is not None:
                    samples.append(result)

        logger.info(f"Successfully loaded {len(samples)} samples")
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


class CTRATEDataset_Submission(MedicalImageDataset):
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

    def _load_data(self, dataset_path):
        self.embedding_dict = {
            p.stem.split("_embedding")[0]: str(p)
            for p in Path(self.embeddings_path).rglob("*_embedding.npz")
        }

        samples = []
        for file_name, embedding_path in tqdm(
            self.embedding_dict.items(), desc="Loading metadata files"
        ):
            image_id = file_name
            sample = {
                "image_id": image_id,
                "embedding_path": embedding_path,
            }
            samples.append(sample)

        return samples

    def get_embedding_path(self, embedding_path):
        """Convert image ID to embedding path"""
        if not hasattr(self, "embedding_dict"):
            logger.error("embedding_dict not found. Ensure _load_data is called first.")
            return os.path.join(str(self.embeddings_path), embedding_path + "_embedding.npz")

        return self.embedding_dict.get(embedding_path, None)

    def extract_sample_list(self, samples):
        """Extract sample list from raw data"""
        extracted_samples = []

        for sample in samples:
            # Report generation sample
            extracted_samples.append(
                {
                    "image_id": sample["image_id"],
                    "image_path": sample["embedding_path"],
                    "question": "Generate a detailed report for this CT scan.",
                    "answer": "",  # sample["tasks"]["report_gen"],
                    "full_answer": "",  # sample["tasks"]["report_gen"],
                    "answer_type": "TEXT",
                    "answer_subtype": "REPORT",
                    "system_text": "You are a medical imaging AI trained to analyze CT scans.",
                }
            )

            extracted_samples.append(
                {
                    "image_id": sample["image_id"],
                    "image_path": sample["embedding_path"],
                    "question": "What is the diagnosis for this CT scan?",
                    "answer": "",  # sample["tasks"]["diagnosis"],
                    "full_answer": "",  # sample["tasks"]["diagnosis"],
                    "answer_type": "TEXT",
                    "answer_subtype": "DIAGNOSIS",
                    "system_text": "You are a medical imaging AI trained to analyze CT scans.",
                }
            )

        # convert to QAItemDict
        extracted_samples = [QAItemDict(sample) for sample in extracted_samples]

        return extracted_samples

    def _filter_samples(self, samples):
        return samples  # No filtering needed for submission dataset
