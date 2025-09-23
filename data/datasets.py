import json
import math
import os
from abc import ABC, abstractmethod
from numbers import Number
from pathlib import Path
from typing import Type

import blosc2
import nrrd
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from utils import logger
from utils.dataset_utils import get_training_image_transforms, preprocess_3d_image
from utils.vision_input_type import VisionInputType

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
    question, answer, model_name, system_text=SYSTEM_TEXT, vision_input_path=None
):

    system_message = {"role": "system", "content": [{"type": "text", "text": system_text}]}

    if model_name == "qwen":
        user_message = {
            "role": "user",
            "content": [
                # TODO: where do we use this vision input path?
                {"type": "video", "vision_input_path": vision_input_path},
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
        dataset_json_path (str): Path to the dataset JSON file
        model_name (str): Name of the VLM the dataset is used for
        split (str): Split name ('train', 'val' or 'test')
        vision_input_path (str | Path): Root path to the vision input (images or embeddings)
        vision_input_type (VisionInputType): Type of vision input (IMAGE or EMBEDDING)
        n_samples (int, optional): Number of samples to load (for debugging)
        answer_types (tuple, optional): Types of qa pairs to include
        split_files (dict, optional): Dictionary with split file paths
        embedding_constant (float or str, optional): If set, use a constant or random embedding
        do_resize (bool, optional): If True, resize images to a fixed size
        do_transforms (bool, optional): If True, apply data augmentations to images
    """

    def __init__(
        self,
        dataset_json_path,
        model_name,
        split="train",
        vision_input_path=None,
        vision_input_type=VisionInputType.EMBEDDING,
        n_samples=None,
        answer_types=("CHOICE",),
        split_files=None,
        embedding_constant=None,
        do_resize=False,
        do_transforms=False
    ):
        super().__init__()

        self.dataset_json_path = Path(dataset_json_path)
        self.vision_input_path = Path(vision_input_path)
        self.vision_input_type = vision_input_type
        self.model_name = model_name
        self.split = split
        self.n_samples = n_samples
        self.split_files = split_files
        self.embedding_constant = embedding_constant
        self.do_resize = do_resize
        self.do_transforms = do_transforms

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

        with open(self.dataset_json_path, "r") as f:
            self.samples = json.load(f)

        self.samples = self._filter_samples(self.samples)

        if self.n_samples is not None:
            self.samples = list(np.random.choice(self.samples, self.n_samples, replace=False))

        if self.do_transforms:
            if self.vision_input_type != VisionInputType.IMAGE:
                raise ValueError("do_transforms can only be True if vision_input_type is IMAGE")
            self.training_transforms = get_training_image_transforms()
        else:
            self.training_transforms = None

    @abstractmethod
    def _get_vision_input_path(self, image_id):
        """Load vision input path for a given image ID"""
        pass

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

    def _load_vision_input(self, vision_input_path:Path):
        """Load vision input from file"""
        try:
            if str(vision_input_path).endswith(".npy"):
                vision_input = np.load(vision_input_path)
            elif str(vision_input_path).endswith(".npz"):
                # Use context manager to ensure proper file closure
                with np.load(vision_input_path) as npz_file:
                    # Attempt to load the "embeddings" key if it exists, otherwise load the first array
                    if "embedding" in npz_file:
                        vision_input = npz_file["embeddings"].copy()  # Make a copy to avoid reference issues
                    else:
                        vision_input = npz_file[npz_file.files[0]].copy()  # Make a copy to avoid reference issues

            elif str(vision_input_path).endswith(".b2nd"):
                # Load b2nd file like nnUNet :)
                dparams = {
                    'nthreads': 1
                }
                mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}
                
                vision_input = np.array(blosc2.open(urlpath=vision_input_path, mode='r', dparams=dparams, **mmap_kwargs))

            elif str(vision_input_path).endswith(".bl2"):
                vision_input = blosc2.load_array(vision_input_path)
                
            elif str(vision_input_path).endswith(".nrrd"):     
                vision_input, header = nrrd.read(vision_input_path)

            elif str(vision_input_path).endswith((".nii.gz", ".nii")):
                vision_input = sitk.GetArrayFromImage(sitk.ReadImage(vision_input_path))          

            else:
                raise ValueError(f"Unsupported vision input file format: {vision_input_path}")
            
            if self.vision_input_type == VisionInputType.IMAGE:
                vision_input = z_score_normalization(vision_input, False, None, np.float32)
                vision_input = preprocess_3d_image(image_array=vision_input, do_resize=self.do_resize, do_transforms=self.do_transforms, split=self.split, training_transforms=self.training_transforms)

            return vision_input
            
        except Exception as e:
            logger.error(f"Error loading vision input from {vision_input_path}: {e}")
            return None

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

        formatted_qa, formatted_qa_prompt_only = format_into_message(
                result["question"],
                result["full_answer"],
                self.model_name,
                system_text=sample["system_text"],
            )

        result["formatted_qa"] = formatted_qa
        result["formatted_qa_prompt_only"] = formatted_qa_prompt_only

        # Load embedding
        vision_input_path = self._get_vision_input_path(sample["image_id"])
        vision_input = self._load_vision_input(vision_input_path)
                
        if self.embedding_constant is not None:
            if type(self.embedding_constant) is float:
                result["vision_input"] = np.full(vision_input.shape, self.embedding_constant, dtype=np.float32)
            elif type(self.embedding_constant) is str and self.embedding_constant == "random":
                result["vision_input"] = np.random.rand(vision_input.shape).astype(np.float32)
            else:
                raise ValueError(f"Unsupported embedding constant type: {type(self.embedding_constant)}")

        if self.vision_input_type == VisionInputType.IMAGE:
            if self.embedding_constant is None:
                result["vision_input"] = vision_input
        else:
            # do we really need this function? why do reshaping and why use torch.tensor()?
            # result["vision_input"] = self._embedding_2_tensor(embd)
            result["vision_input"] = vision_input

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
        split (str): Split name ('train', 'val' or 'test')
        vision_input_path (str | Path): Root path to the vision input (images or embeddings)
        vision_input_type (VisionInputType): Type of vision input (IMAGE or EMBEDDING)
        n_samples (int, optional): Number of samples to load (for debugging)
        answer_types (tuple, optional): Types of qa pairs to include
        split_files (dict, optional): Dictionary with split file paths
        embedding_constant (float or str, optional): If set, use a constant or random embedding
        do_resize (bool, optional): If True, resize images to a fixed size
        do_transforms (bool, optional): If True, apply data augmentations to images
    """
    def _get_vision_input_path(self, image_id):
        """Load vision input path for a given image ID"""
        if self.split == "train":
            image_name = f"{image_id.split("./")[-1].split(".")[0]}_0000"
        else:
            image_name = f"{image_id.split("./")[-1].split(".")[0]}"
        if self.vision_input_type == VisionInputType.EMBEDDING:
            image_name = f"{image_name}_embedding.bl2"
        elif self.vision_input_type == VisionInputType.IMAGE:
            image_name = f"{image_name}.b2nd"
        else:
            raise ValueError(f"Unsupported vision input type: {self.vision_input_type}")
        return str(self.vision_input_path / image_name)


class CTRATEDataset(MedicalImageDataset):
    """
    CT-RATE dataset class for CT scan analysis

    Args:
        dataset_json_path (str): Path to the dataset JSON file
        model_name (str): Name of the VLM the dataset is used for
        split (str): Split name ('train', 'val' or 'test')
        vision_input_path (str | Path): Root path to the vision input (images or embeddings)
        vision_input_type (VisionInputType): Type of vision input (IMAGE or EMBEDDING)
        n_samples (int, optional): Number of samples to load (for debugging)
        answer_types (tuple, optional): Types of qa pairs to include
        split_files (dict, optional): Dictionary with split file paths
        embedding_constant (float or str, optional): If set, use a constant or random embedding
        do_resize (bool, optional): If True, resize images to a fixed size
        do_transforms (bool, optional): If True, apply data augmentations to images
    """
    def _get_vision_input_path(self, image_id):
        """Load vision input path for a given image ID"""
        split_path = "valid" if self.split == "val" else self.split
        if self.vision_input_type == VisionInputType.EMBEDDING:
            return Path(self.vision_input_path / f"{split_path}_fixed" / f"{image_id}_embedding.bl2")
        if self.vision_input_type == VisionInputType.IMAGE:
            return Path(self.vision_input_path / f"{split_path}_fixed" / f"{image_id}.b2nd")
        else:
            raise ValueError(f"Unsupported vision input type: {self.vision_input_type}")
        
