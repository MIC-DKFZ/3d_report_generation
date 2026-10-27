"""CT-RATE paths, sampling, and report augmentation."""

import random
import re
from pathlib import Path
from typing import ClassVar

import numpy as np

from vlm.data.paths import resolve_vision_directory
from vlm.data.types import VisionInputType
from vlm.utils import logger

from .base import MedicalImageDataset


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
        do_resize (bool, optional): If True, resize images to a fixed size
        do_transforms (bool, optional): If True, apply data augmentations to images
        vision_encoder_patch_size (tuple, optional): Patch size of the vision encoder
    """

    dataset_name: ClassVar[str] = "ctrate"

    @staticmethod
    def _base_image_id(image_id: str) -> str:
        return image_id.rsplit("_", 1)[0]

    def _select_samples(self, samples: list[dict]) -> list[dict]:
        if self.n_samples != "unique":
            return super()._select_samples(samples)

        grouped_indices: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            base_image_id = self._base_image_id(sample["image_id"])
            grouped_indices.setdefault(base_image_id, []).append(index)

        # Use the same local deterministic RNG pattern as integer n_samples selection.
        rng = np.random.default_rng(42)
        selected_indices = [int(rng.choice(indices)) for indices in grouped_indices.values()]

        logger.info_main(
            f"Selected {len(selected_indices)} unique CT-RATE images from "
            f"{len(samples)} samples for {self.__class__.__name__} ({self.split})."
        )
        return [samples[i] for i in selected_indices]

    def _get_vision_input_path(self, image_id, split_path=None) -> Path:
        """Resolve an input path, optionally overriding the split for retrieved neighbors."""
        directory = resolve_vision_directory(
            self.vision_input_path, self.dataset_name, split_path or self.split
        )

        if self.vision_input_type == VisionInputType.EMBEDDING:
            return directory / f"{image_id}{self.image_id_suffix}_embedding.bl2"
        if self.vision_input_type == VisionInputType.IMAGE:
            return directory / f"{image_id}.b2nd"
        else:
            raise ValueError(f"Unsupported vision input type: {self.vision_input_type}")

    def shuffle_sentences(self, question: str, answer: str) -> tuple[str, str]:
        """
        Randomly shuffle sentences within Findings and Impression sections separately
        """
        if answer is None or not str(answer).strip():
            return question, ""

        answer_str = str(answer).strip()
        # Split by headers, keeping them in the result
        sections = re.split(r"(Findings:|Impression:)", answer_str)

        result = []
        for part in sections:
            # Keep headers as-is
            if part in ["Findings:", "Impression:"]:
                result.append(part)
            # Shuffle sentences in content sections
            elif part.strip():
                sentences = re.split(r"(?<=[.!?])\s+", part.strip())
                if len(sentences) > 1 and random.uniform(0, 1) < self.sentence_shuffle_p:
                    random.shuffle(sentences)
                    result.append(" " + " ".join(sentences))
                else:
                    result.append(part)

        return question, ''.join(result)
