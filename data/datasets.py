import json
import math
import os
from abc import ABC, abstractmethod
from numbers import Number
from pathlib import Path
from typing import Type, Literal
import re
import random
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
        embedding_constant (float or str, optional): If set, use a constant or random embedding
        do_resize (bool, optional): If True, resize images to a fixed size
        do_transforms (bool, optional): If True, apply data augmentations to images
        vision_encoder_patch_size (tuple, optional): Patch size of the vision encoder
    """

    def __init__(
        self,
        dataset_json_path,
        model_name,
        split: Literal["train", "val", "test"] = "train",
        vision_input_path=None,
        vision_input_type=VisionInputType.EMBEDDING,
        n_samples=None,
        answer_types=("CHOICE",),
        embedding_constant=None,
        do_resize=False,
        do_transforms=False,
        vision_encoder_patch_size=None,
        do_sentence_shuffle=False
    ):
        super().__init__()

        self.dataset_json_path = Path(dataset_json_path)
        self.vision_input_path = Path(vision_input_path)
        self.vision_input_type = vision_input_type
        self.model_name = model_name
        self.split = split
        self.n_samples = n_samples
        self.embedding_constant = embedding_constant
        self.do_resize = do_resize
        self.do_transforms = do_transforms
        self.vision_encoder_patch_size = vision_encoder_patch_size
        self.do_sentence_shuffle = do_sentence_shuffle

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
            self.training_transforms = get_training_image_transforms(patch_size=self.vision_encoder_patch_size)
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
            vision_input = preprocess_3d_image(image_array=vision_input, do_resize=self.do_resize, do_transforms=self.do_transforms, patch_size=self.vision_encoder_patch_size, split=self.split, training_transforms=self.training_transforms)

        return vision_input

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

        if self.do_sentence_shuffle:
            result["question"], result["full_answer"] = self.shuffle_sentences(result["question"], result["full_answer"])
        else:
            result["full_answer"] = result["full_answer"]

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
                result["vision_input"] = np.random.rand(*vision_input.shape).astype(np.float32)
            else:
                raise ValueError(f"Unsupported embedding constant type: {type(self.embedding_constant)}")
        else:
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

    def shuffle_sentences(self, question, answer):
        # Needs to be implemented for each dataset
        return question, answer


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
        embedding_constant (float or str, optional): If set, use a constant or random embedding
        do_resize (bool, optional): If True, resize images to a fixed size
        do_transforms (bool, optional): If True, apply data augmentations to images
        vision_encoder_patch_size (tuple, optional): Patch size of the vision encoder
    """
    def _get_vision_input_path(self, image_id):
        """Load vision input path for a given image ID"""
        if self.split == "train":
            image_name = f"{image_id.split("./")[-1].split(".")[0]}_0000"
        elif self.split == "val":
            image_name = f"{image_id.split("./")[-1].split(".")[0]}_0000"
        elif self.split == "test":
            image_name = f"{image_id.split("./")[-1].split(".")[0]}"
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if self.vision_input_type == VisionInputType.EMBEDDING:
            image_name = f"{image_name}_embedding.bl2"
        elif self.vision_input_type == VisionInputType.IMAGE:
            image_name = f"{image_name}.b2nd"
        else:
            raise ValueError(f"Unsupported vision input type: {self.vision_input_type}")
        return str(self.vision_input_path / image_name)

    def shuffle_sentences(self, question, answer) -> tuple[str, str]:
        # Extract options with their content
        option_pattern = r'^([A-D])\.\s*(.+)$'
        lines = question.split('\n')
        
        options = {}
        other_lines = []
        
        for line in lines:
            match = re.match(option_pattern, line)
            if match:
                options[match.group(1)] = match.group(2)
            else:
                other_lines.append(line)
        
        if len(options) != 4:
            return question, answer
        
        # Find current correct answer letter
        answer_match = re.search(r'(?:The answer is )?([A-D])\.?', answer, re.IGNORECASE)
        if not answer_match:
            return question, answer
        
        old_letter = answer_match.group(1)
        correct_content = options[old_letter]
        
        # Shuffle the content
        contents = list(options.values())
        random.shuffle(contents)
        
        # Rebuild options A, B, C, D with shuffled content
        new_options = {chr(ord('A') + i): contents[i] for i in range(4)}
        
        # Find new letter for correct content
        new_letter = [k for k, v in new_options.items() if v == correct_content][0]
        
        # Rebuild question
        option_lines = [f"{letter}. {new_options[letter]}" for letter in ['A', 'B', 'C', 'D']]
        new_question = '\n'.join(other_lines + option_lines)
        
        # Update answer letter
        new_answer = re.sub(
            r'((?:The answer is )?)([A-D])(\.?)',
            lambda m: m.group(1) + new_letter + m.group(3),
            answer,
            count=1,  # only replace the first match
            flags=re.IGNORECASE
        )
        
        return new_question, new_answer


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
        embedding_constant (float or str, optional): If set, use a constant or random embedding
        do_resize (bool, optional): If True, resize images to a fixed size
        do_transforms (bool, optional): If True, apply data augmentations to images
        vision_encoder_patch_size (tuple, optional): Patch size of the vision encoder
    """
    def _get_vision_input_path(self, image_id):
        """Load vision input path for a given image ID"""
        if self.split == "train":
            split_path = "train"
        elif self.split == "val":
            split_path = "train"
        elif self.split == "test":
            split_path = "valid"
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if self.vision_input_type == VisionInputType.EMBEDDING:
            return Path(self.vision_input_path / f"{split_path}_fixed" / f"{image_id}_embedding.bl2")
        if self.vision_input_type == VisionInputType.IMAGE:
            return Path(self.vision_input_path / f"{split_path}_fixed" / f"{image_id}.b2nd")
        else:
            raise ValueError(f"Unsupported vision input type: {self.vision_input_type}")
        
    def shuffle_sentences(self, question: str, answer: str, p: float = 0.5) -> tuple[str, str]:
        """
        Randomly shuffle sentences within Findings and Impression sections separately
        """
        if answer is None or not str(answer).strip():
            return question, ""
        
        answer_str = str(answer).strip()
        if not self.shuffle_sentences:
            return question, answer_str
        
        # Split by headers, keeping them in the result
        sections = re.split(r'(Findings:|Impression:)', answer_str)
        
        result = []
        for part in sections:
            # Keep headers as-is
            if part in ['Findings:', 'Impression:']:
                result.append(part)
            # Shuffle sentences in content sections
            elif part.strip():
                sentences = re.split(r'(?<=[.!?])\s+', part.strip())
                if len(sentences) > 1 and random.uniform(0, 1) < p:
                    random.shuffle(sentences)
                    result.append(' ' + ' '.join(sentences))
                else:
                    result.append(part)

        return question, ''.join(result)
