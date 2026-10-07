"""Shared sample selection, message formatting, and input preparation."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
from torch.utils.data import Dataset

from vlm.configs.run_config import RunConfig
from vlm.data.embedding_transforms import EmbeddingCompose
from vlm.utils import logger
from vlm.utils.data.dataset_utils import get_training_image_transforms
from vlm.utils.data.vision_input_type import VisionInputType

from .vision import load_vision_input

SYSTEM_TEXT = "You are a medical imaging AI trained to analyze CT scans."


class MedicalImageDataset(Dataset, ABC):
    """
    Base class for medical imaging datasets with vision-language models

    Args:
        dataset_json_path (str): Path to the dataset JSON file
        split (str): Split name ('train', 'val' or 'test')
        vision_input_path (str | Path): Root path to the vision input (images or embeddings)
        vision_input_type (VisionInputType): Type of vision input (IMAGE or EMBEDDING)
        n_samples (int, optional): Number of samples to load (for debugging)
        answer_types (tuple, optional): Types of qa pairs to include
        do_resize (bool, optional): If True, resize images to a fixed size
        do_transforms (bool, optional): If True, apply data augmentations to images
        vision_encoder_patch_size (tuple, optional): Patch size of the vision encoder
    """

    def __init__(
        self,
        cfg: RunConfig,
        split: Literal["train", "val", "test"],
        vision_encoder_patch_size: Tuple[int, ...],
        n_samples: int | Literal["unique"] | None = None,
    ):
        super().__init__()

        self.n_samples = n_samples

        self.vision_input_path = Path(cfg.data.vision_input_path)
        self.vision_input_type = cfg.data.vision_input_type
        self.split = split
        self.text_only = cfg.data.text_only
        self.do_resize = cfg.data.do_resize
        self.do_transforms = cfg.data.do_transforms and self.split == "train"
        self.vision_encoder_patch_size = vision_encoder_patch_size
        self.is_stacked_2d_embedding = cfg.data.is_stacked_2d_embedding
        self.reduce_to_n_slices = cfg.data.reduce_to_n_slices
        self.embedding_transforms = self._build_embedding_transforms(cfg)

        if self.split == "train":
            self.dataset_json_path = Path(cfg.data.train_dataset_json_path)
        elif self.split == "val":
            self.dataset_json_path = Path(cfg.data.val_dataset_json_path)
        elif self.split == "test":
            self.dataset_json_path = Path(cfg.data.test_dataset_json_path)

        logger.info_main(
            f"Using vision_input_path for {self.__class__.__name__} ({self.split}): {self.vision_input_path}"
        )

        self.sentence_shuffle_p = cfg.data.sentence_shuffle_p if self.split == "train" else 0.0
        self.image_id_suffix = cfg.data.image_id_suffix

        answer_types = cfg.data.answer_types

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

        self.samples = self._select_samples(self.samples)

        if self.do_transforms:
            if self.vision_input_type != VisionInputType.IMAGE:
                raise ValueError("do_transforms can only be True if vision_input_type is IMAGE")
            self.training_transforms = get_training_image_transforms(
                patch_size=self.vision_encoder_patch_size
            )
        else:
            self.training_transforms = None

    def _build_embedding_transforms(self, cfg: RunConfig) -> EmbeddingCompose | None:
        if self.split != "train" or self.vision_input_type != VisionInputType.EMBEDDING:
            return None

        transforms = [transform_cfg.to_instance() for transform_cfg in cfg.data.embedding_transforms]
        if not transforms:
            return None

        return EmbeddingCompose(transforms)

    @abstractmethod
    def _get_vision_input_path(self, image_id) -> Path:
        """Load vision input path for a given image ID"""
        pass

    def _select_samples(self, samples: list[dict]) -> list[dict]:
        if self.n_samples is None:
            return samples

        if not isinstance(self.n_samples, int):
            raise ValueError(
                f"Unsupported n_samples={self.n_samples!r} for {self.__class__.__name__}. "
                "MedicalImageDataset only supports integer n_samples; subclasses must "
                "override _select_samples to support custom sampling modes."
            )

        if self.n_samples > len(samples):
            logger.warning_main(
                f"Requested n_samples={self.n_samples} for {self.__class__.__name__} "
                f"({self.split}), but only {len(samples)} samples are available."
            )
            return samples

        if self.n_samples < len(samples):
            # Use a local RNG so this subset does not depend on earlier random draws.
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(len(samples), self.n_samples, replace=False)
            return [samples[i] for i in sample_indices]

        return samples

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

    def _load_vision_input(self, vision_input_path: Path):
        return load_vision_input(
            vision_input_path,
            vision_input_type=self.vision_input_type,
            do_resize=self.do_resize,
            do_transforms=self.do_transforms,
            patch_size=self.vision_encoder_patch_size,
            split=self.split,
            training_transforms=self.training_transforms,
            is_stacked_2d_embedding=self.is_stacked_2d_embedding,
            reduce_to_n_slices=self.reduce_to_n_slices,
        )

    def __getitem__(self, index):
        sample = self.samples[index]

        # Create a new dictionary instead of modifying the stored one
        result = dict(sample)

        # Alter question
        # result = self.modify_sample(result)

        if self.sentence_shuffle_p > 0:
            result["question"], result["full_answer"] = self.shuffle_sentences(
                result["question"], result["full_answer"]
            )

        formatted_qa, formatted_qa_prompt_only = self.format_into_message(
            result["question"],
            result["full_answer"],
            system_text=sample["system_text"],
        )

        result["formatted_qa"] = formatted_qa
        result["formatted_qa_prompt_only"] = formatted_qa_prompt_only

        if not self.text_only:
            result = self._load_embeddings(sample, result)
        else:
            result["vision_input"] = None
        return result

    def _load_embeddings(self, sample, result: Dict) -> Dict:
        vision_input_path = self._get_vision_input_path(sample["image_id"])
        vision_input = self._load_vision_input(vision_input_path)

        if self.embedding_transforms is not None:
            vision_input = self.embedding_transforms(vision_input)
        result["vision_input"] = vision_input

        return result

    def __len__(self):
        return len(self.samples)

    def verify(self):
        """Test function to verify dataset loading"""
        logger.info_main(f"\nTotal number of samples: {len(self)}")
        logger.info_main("\nExample question:")
        if len(self.samples) > 0:
            sample = self.samples[0]
            logger.info_main(f"Image ID: {sample['image_id']}")
            logger.info_main(f"Question: {sample['question']}")
            logger.info_main(f"Answer: {sample['full_answer']}")
            logger.info_main(f"Answer Type: {sample['answer_type']}")
            logger.info_main(f"Answer Subtype: {sample['answer_subtype']}")

    def shuffle_sentences(self, question, answer):
        # Needs to be implemented for each dataset
        return question, answer

    def format_into_message(self, question, answer, system_text=SYSTEM_TEXT):
        """Build model-neutral messages; wrappers translate `vision` items for their model."""
        system_message = {"role": "system", "content": [{"type": "text", "text": system_text}]}
        user_message = {
            "role": "user",
            "content": ([{"type": "vision"}] if not self.text_only else [])
            + [{"type": "text", "text": question}],
        }
        answer_message = {"role": "assistant", "content": [{"type": "text", "text": answer}]}

        full_conversation = [system_message, user_message, answer_message]
        prompt_only_conversation = [system_message, user_message]
        return full_conversation, prompt_only_conversation
