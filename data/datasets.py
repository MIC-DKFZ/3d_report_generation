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
        """Load embedding path for a given image ID"""
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

    def _load_embedding(self, embedding_path:Path):
        """Load embedding from file"""
        try:
            if str(embedding_path).endswith(".npy"):
                embedding = np.load(embedding_path)
            elif str(embedding_path).endswith(".npz"):
                # Use context manager to ensure proper file closure
                with np.load(embedding_path) as npz_file:
                    # Attempt to load the "embeddings" key if it exists, otherwise load the first array
                    if "embedding" in npz_file:
                        embedding = npz_file["embeddings"].copy()  # Make a copy to avoid reference issues
                    else:
                        embedding = npz_file[npz_file.files[0]].copy()  # Make a copy to avoid reference issues

            elif str(embedding_path).endswith(".b2nd"):
                # Load b2nd file like nnUNet :)
                dparams = {
                    'nthreads': 1
                }
                mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}
                
                embedding = np.array(blosc2.open(urlpath=embedding_path, mode='r', dparams=dparams, **mmap_kwargs))
                
            elif str(embedding_path).endswith(".nrrd"):     
                embedding, header = nrrd.read(embedding_path)

            elif str(embedding_path).endswith((".nii.gz", ".nii")):
                embedding = sitk.GetArrayFromImage(sitk.ReadImage(embedding_path))
                embedding = self.mask_crop_embedding(embedding=embedding, embedding_path=embedding_path)               

            else:
                raise ValueError(f"Unsupported embedding file format: {embedding_path}")
            
            if self.vision_input_type == VisionInputType.IMAGE:
                embedding = z_score_normalization(embedding, False, None, np.float32)
                embedding = preprocess_3d_image(image_array=embedding, do_resize=self.do_resize, do_transforms=self.do_transforms, split=self.split, training_transforms=self.training_transforms)

            return embedding
            
        except Exception as e:
            logger.error(f"Error loading embedding from {embedding_path}: {e}")
            return None

    def mask_crop_embedding(self, embedding, embedding_path):
        """Crop embedding to a region defined by a segmentation mask"""
        return embedding

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
        embd = self._load_embedding(vision_input_path)
                
        if self.embedding_constant is not None:
            if type(self.embedding_constant) is float:
                result["embedding"] = np.full(embd.shape, self.embedding_constant, dtype=np.float32)
            elif type(self.embedding_constant) is str and self.embedding_constant == "random":
                result["embedding"] = np.random.rand(embd.shape).astype(np.float32)
            else:
                raise ValueError(f"Unsupported embedding constant type: {type(self.embedding_constant)}")

        if self.vision_input_type == VisionInputType.IMAGE:
            if self.embedding_constant is None:
                result["embedding"] = embd
        else:
            # do we really need this function? why do reshaping and why use torch.tensor()?
            result["embedding"] = self._embedding_2_tensor(result["embedding"])

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
    def _get_vision_input_path(self, image_id):
        """Load embedding path for a given image ID"""
        if self.vision_input_type == VisionInputType.IMAGE:
            # TODO: image_id is in format ./imagesTr/amosxxx_0000.nii.gz
            # Do we want to keep this format or change it to amosxxx?
            if self.split == "train":
                image_name = f"{image_id.split("./")[-1].split(".")[0]}_0000.nii.gz"
            else:
                image_name = f"{image_id.split("./")[-1].split(".")[0]}.nii.gz"
            return str(self.vision_input_path / image_name)
        # TODO: adapt for embedding input type
        return str(self.vision_input_path / f"{image_id}_embedding.npy")

    def get_embedding_path(self, image_path):
        """Convert image path to embedding path"""
        base_name = image_path.replace(".nii.gz", "")
        return str(os.path.join(self.vision_input_path, f"{base_name}_embedding.npy"))


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
        use_csv_reports (bool, optional): If True, load from CSV files instead of YML files
    """
    def _get_vision_input_path(self, image_id):
        """Load embedding path for a given image ID"""
        if self.vision_input_type == VisionInputType.IMAGE:
            parts = image_id.split('_')
            patient_folder = '_'.join(parts[:-2])  # "valid_1"
            patient_id = '_'.join(parts[:-1])      # "valid_1_a"
            split_path = "valid" if self.split == "val" else self.split
            return Path(self.vision_input_path / f"{split_path}_fixed" / patient_folder / patient_id / f"{image_id}.nii.gz")
        # TODO: adapt for embedding input type
        return str(self.vision_input_path / f"{image_id}.nii.gz")

    def mask_crop_embedding(self, embedding, embedding_path):
        embedding_path = str(embedding_path)
        lung_segmentation_path = None
        if "valid_fixed" in embedding_path:
            lung_segmentation_path = embedding_path.replace("valid_fixed", "ts_seg/ts_total/valid_fixed")
        if "train_fixed" in embedding_path:
            lung_segmentation_path = embedding_path.replace("train_fixed", "ts_seg/ts_total/train_fixed")

        if os.path.exists(lung_segmentation_path):
            # we have no cropped images on dkfz drives, therefore this if block is here. This is only used for prototyping and is not io efficient.
            # use with:
            # "--dataset_path",
            # "/mnt/e230-thrp-data/mic_rocket/data/raw/CTRATEv2/CT-RATE/dataset",
            # "--embeddings_path",
            # "/mnt/e230-thrp-data/mic_rocket/data/raw/CTRATEv2/CT-RATE/dataset/valid_fixed",
            lung_segmentation = sitk.GetArrayFromImage(sitk.ReadImage(lung_segmentation_path))

            lung_indices = [10, 11, 12, 13, 14]
            # Create a mask for the lung regions
            lung_mask = np.isin(lung_segmentation, lung_indices)
            # Crop the embedding to the bounding box of the lung mask with 4% offset
            coords = np.argwhere(lung_mask)

            # same computation as Stefan used for Dataset802_LungCrop on juwels
            if coords.size > 0:
                min_coords = coords.min(axis=0)
                max_coords = coords.max(axis=0) + 1  # +1 because slicing is exclusive
                
                # Calculate dimensions and 4% expansion (scale/2 on each side)
                dx, dy, dz = max_coords - min_coords
                ex, ey, ez = dx * 0.02, dy * 0.02, dz * 0.02  # 0.04/2 = 0.02 for each side
                
                # Apply expansion with bounds checking using floor/ceil
                new_min_x = max(0, math.floor(min_coords[0] - ex))
                new_max_x = min(embedding.shape[0], math.ceil(max_coords[0] + ex))
                new_min_y = max(0, math.floor(min_coords[1] - ey))
                new_max_y = min(embedding.shape[1], math.ceil(max_coords[1] + ey))
                new_min_z = max(0, math.floor(min_coords[2] - ez))
                new_max_z = min(embedding.shape[2], math.ceil(max_coords[2] + ez))
                
                embedding = embedding[new_min_x:new_max_x, new_min_y:new_max_y, new_min_z:new_max_z]

                print("embeddings shape after lung crop", embedding.shape)
        
        return embedding
        # viewer = napari.Viewer()
        # viewer.add_image(embedding, name="CT Patch")
        # viewer.add_image(embedding_full, name="CT Full")
        # napari.run()
        
