import json
import os
from abc import ABC, abstractmethod
from numbers import Number
from pathlib import Path
from typing import Type, Literal, Tuple, Dict, List
import re
import random
import blosc2
import nrrd
import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange
from torch.utils.data import Dataset

from vlm.data.embedding_transforms import EmbeddingCompose
from vlm.utils import logger
from vlm.utils.dataset_utils import get_training_image_transforms, preprocess_3d_image
from vlm.utils.regex_patterns import (
    VQA_ANSWER_PATTERN,
    VQA_OPTION_PATTERN,
    VQA_OPTION_LETTERS_IN_TEXT_PATTERN,
)
from vlm.utils.vision_input_type import VisionInputType
from vlm.models.phi4mini.wrapper import PHI_VISION_PLACEHOLDER_TOKEN
from vlm.configs.run_config import RunConfig

SYSTEM_TEXT = "You are a medical imaging AI trained to analyze CT scans."


def convert_dtype(image: np.ndarray, target_dtype: Type[Number]) -> np.ndarray:
    return image.astype(target_dtype)


def z_score_normalization(
    image: np.ndarray,
    use_mask_for_norm: bool,
    non_zero_mask: np.ndarray,
    target_dtype: Type[Number],
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


class MedicalImageDataset(Dataset, ABC):
    """
    Base class for medical imaging datasets with vision-language models

    Args:
        dataset_json_path (str): Path to the dataset JSON file
        model_type (str): Name of the VLM the dataset is used for
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
        cfg: RunConfig,
        split: Literal["train", "val", "test"],
        vision_encoder_patch_size: Tuple[int, ...],
        n_samples: int | None = None,
    ):
        super().__init__()

        self.n_samples = n_samples

        self.vision_input_path = Path(cfg.data.vision_input_path)
        self.vision_input_type = cfg.data.vision_input_type
        self.model_type = cfg.model.model_type
        self.split = split
        self.text_only = cfg.data.text_only
        self.embedding_constant = cfg.data.embedding_constant
        self.do_resize = cfg.data.do_resize
        self.do_transforms = cfg.data.do_transforms
        self.vision_encoder_patch_size = vision_encoder_patch_size
        self.is_stacked_2d_embedding = cfg.data.is_stacked_2d_embedding
        self.reduce_to_n_slices = cfg.data.reduce_to_n_slices
        self.embedding_transforms = self._build_embedding_transforms(cfg)

        if self.split == "train":
            self.dataset_json_path = Path(cfg.data.train_dataset_json_path)
            self.do_sentence_shuffle = cfg.data.do_sentence_shuffle
        elif self.split == "val":
            self.dataset_json_path = Path(cfg.data.val_dataset_json_path)
            self.do_sentence_shuffle = False
        elif self.split == "test":
            self.dataset_json_path = Path(cfg.eval.test_dataset_json_path)
            self.do_sentence_shuffle = False

        logger.info_main(
            f"Using vision_input_path for {self.__class__.__name__} ({self.split}): {self.vision_input_path}"
        )

        self.sentence_shuffle_p = cfg.data.sentence_shuffle_p
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
        """Load vision input from file"""
        if str(vision_input_path).endswith(".npy"):
            vision_input = np.load(vision_input_path)
        elif str(vision_input_path).endswith(".npz"):
            # Use context manager to ensure proper file closure
            with np.load(vision_input_path) as npz_file:
                # Attempt to load the "embeddings" key if it exists, otherwise load the first array
                if "embedding" in npz_file:
                    vision_input = npz_file[
                        "embeddings"
                    ].copy()  # Make a copy to avoid reference issues
                else:
                    vision_input = npz_file[
                        npz_file.files[0]
                    ].copy()  # Make a copy to avoid reference issues

        elif str(vision_input_path).endswith(".b2nd"):
            # Load b2nd file like nnUNet :)
            dparams = {"nthreads": 1}
            mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}

            vision_input = np.array(
                blosc2.open(urlpath=vision_input_path, mode="r", dparams=dparams, **mmap_kwargs)
            )

        elif str(vision_input_path).endswith(".bl2"):
            # remove batch dimension
            vision_input = blosc2.load_array(vision_input_path)

            if vision_input.ndim == 5 and vision_input.shape[0] == 1:
                vision_input = vision_input.squeeze(0)

        elif str(vision_input_path).endswith(".nrrd"):
            vision_input, header = nrrd.read(vision_input_path)

        elif str(vision_input_path).endswith((".nii.gz", ".nii")):
            vision_input = sitk.GetArrayFromImage(sitk.ReadImage(vision_input_path))

        else:
            raise ValueError(f"Unsupported vision input file format: {vision_input_path}")

        if self.vision_input_type == VisionInputType.IMAGE:
            vision_input = z_score_normalization(vision_input, False, None, np.float32)
            vision_input = preprocess_3d_image(
                image_array=vision_input,
                do_resize=self.do_resize,
                do_transforms=self.do_transforms,
                patch_size=self.vision_encoder_patch_size,
                split=self.split,
                training_transforms=self.training_transforms,
            )

        if self.is_stacked_2d_embedding:
            # remove class token
            vision_input = vision_input[:, 1:, :]

            n_slices, n_patch_tokens, embed_dim = vision_input.shape
            n_side = int(n_patch_tokens ** 0.5)

            vision_input = rearrange(vision_input, "s (h w) c -> s h w c", h=n_side, w=n_side)

            bins = np.linspace(0, n_slices, self.reduce_to_n_slices + 1).astype(int)

            # do 'average pooling' to compress slice dimension
            pooled_slices = [
                vision_input[bins[i]:bins[i + 1]].mean(axis=0)
                for i in range(self.reduce_to_n_slices)
            ]
            vision_input = np.stack(pooled_slices, axis=0)
            vision_input = rearrange(vision_input, "s h w c -> c s h w")

        elif vision_input.ndim == 3:
            # add channel dim if missing
            vision_input = vision_input[None, ...]

        return vision_input

    def _embedding_2_tensor(self, embedding):
        """Convert embedding to tensor"""
        # TODO check if we can remove this function!
        if embedding is None:
            logger.error_main("Embedding is None, cannot convert to tensor.")
            return None

        np_img_flat = embedding.reshape(embedding.shape[0], embedding.shape[1], -1)
        np_img_flat = np_img_flat.transpose(0, 2, 1)
        img_tensor = torch.tensor(np_img_flat)

        return img_tensor

    # def modify_sample(self, result: Dict) -> Dict:
    #     return result

    def __getitem__(self, index):
        sample = self.samples[index]

        # Create a new dictionary instead of modifying the stored one
        result = dict(sample)

        # Alter question
        # result = self.modify_sample(result)

        if self.do_sentence_shuffle:
            result["question"], result["full_answer"] = self.shuffle_sentences(
                result["question"], result["full_answer"]
            )
        else:
            result["full_answer"] = result["full_answer"]

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

        if self.embedding_constant is not None:
            if type(self.embedding_constant) is float:
                result["vision_input"] = np.full(
                    vision_input.shape, self.embedding_constant, dtype=np.float32
                )
            elif type(self.embedding_constant) is str and self.embedding_constant == "random":
                result["vision_input"] = np.random.rand(*vision_input.shape).astype(np.float32)
            else:
                raise ValueError(
                    f"Unsupported embedding constant type: {type(self.embedding_constant)}"
                )
        else:
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

    def format_into_message(
            self, question, answer, system_text=SYSTEM_TEXT
    ):

        if "qwen" in self.model_type.lower():
            system_message = {"role": "system", "content": [{"type": "text", "text": system_text}]}
            user_message = {
                "role": "user",
                "content": ([{"type": "video"}] if not self.text_only else [])
                + [{"type": "text", "text": question}],
            }
            answer_message = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        elif "gemma" in self.model_type.lower():
            system_message = {"role": "system", "content": [{"type": "text", "text": system_text}]}
            user_message = {
                "role": "user",
                "content": ([{"type": "image"}] if not self.text_only else [])
                + [{"type": "text", "text": question}],
            }
            answer_message = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        elif "phi" in self.model_type.lower():
            system_message = {"role": "system", "content": system_text}
            vision_prefix = f"{PHI_VISION_PLACEHOLDER_TOKEN}\n" if not self.text_only else ""
            user_message = {"role": "user", "content": f"{vision_prefix}{question}"}
            answer_message = {"role": "assistant", "content": answer}
        else:
            raise ValueError(f"Unsupported model name: {self.model_type}")

        full_conversation = [system_message, user_message, answer_message]
        prompt_only_conversation = [system_message, user_message]
        return full_conversation, prompt_only_conversation


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

    def _get_vision_input_path(self, image_id) -> Path:
        """Load vision input path for a given image ID"""
        if self.split == "train":
            image_name = f"{image_id.split('./')[-1].split('.')[0]}_0000"
        elif self.split == "val":
            image_name = f"{image_id.split('./')[-1].split('.')[0]}_0000"
        elif self.split == "test":
            image_name = f"{image_id.split('./')[-1].split('.')[0]}"
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if self.vision_input_type == VisionInputType.EMBEDDING:
            image_name = f"{image_name}_embedding.bl2"
        elif self.vision_input_type == VisionInputType.IMAGE:
            image_name = f"{image_name}.b2nd"
        else:
            raise ValueError(f"Unsupported vision input type: {self.vision_input_type}")
        return Path(self.vision_input_path / image_name)

    def shuffle_sentences(self, question, answer) -> tuple[str, str]:
        # Extract options with their content
        lines = question.split("\n")

        options = {}
        lines_before = []
        lines_after = []

        # There are always exactly 4 options
        for line in lines:
            if len(options) < 4:
                match = re.match(VQA_OPTION_PATTERN, line)
                if match:
                    options[match.group(1)] = match.group(2)
                else:
                    lines_before.append(line)
            else:
                lines_after.append(line)

        if len(options) != 4:
            return question, answer

        # Find current correct answer letter
        # answer_match = re.search(VQA_ANSWER_PATTERN, answer)
        # if not answer_match:
        #     return question, answer

        # old_letter = answer_match.group(2)
        # correct_content = options[old_letter]

        # Shuffle the content
        contents = list(options.values())
        random.shuffle(contents)

        # Rebuild options A, B, C, D with shuffled content
        new_options = {chr(ord("A") + i): contents[i] for i in range(4)}

        # Find new letter for correct content
        # new_letter = [k for k, v in new_options.items() if v == correct_content][0]

        # Rebuild question
        option_lines = [f"{letter}. {new_options[letter]}" for letter in ["A", "B", "C", "D"]]
        new_question = "\n".join(lines_before + option_lines + lines_after)

        # Update answer letter
        # new_answer = re.sub(
        #     VQA_ANSWER_PATTERN,
        #     lambda m: (m.group(1) or "") + new_letter + (m.group(3) or ""),
        #     answer,
        #     count=1  # only replace the first match
        # )

        # Update any option letters by their new ones in the answer
        reversed_new_options = {content: letter for letter, content in new_options.items()}
        options_letter_map = {}
        for old_letter, content in options.items():
            options_letter_map[old_letter] = reversed_new_options[
                content
            ]  # map old letter to new letter

        new_answer = re.sub(
            VQA_OPTION_LETTERS_IN_TEXT_PATTERN,
            lambda m: options_letter_map[
                next(g for g in m.groups() if g)
            ],  # replace old letter with new letter
            answer,
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
        """Load vision input path for a given image ID"""
        if not split_path:
            if self.split == "train":
                split_path = "train"
            elif self.split == "val":
                split_path = "train"
            elif self.split == "test":
                split_path = "valid"
            else:
                raise ValueError(f"Unknown split: {self.split}")

        if self.vision_input_type == VisionInputType.EMBEDDING:
            return Path(
                self.vision_input_path
                / f"{split_path}_fixed"
                / f"{image_id}{self.image_id_suffix}_embedding.bl2"
            )
        if self.vision_input_type == VisionInputType.IMAGE:
            return Path(self.vision_input_path / f"{split_path}_fixed" / f"{image_id}.b2nd")
        else:
            raise ValueError(f"Unsupported vision input type: {self.vision_input_type}")

    def shuffle_sentences(self, question: str, answer: str) -> tuple[str, str]:
        """
        Randomly shuffle sentences within Findings and Impression sections separately
        """
        if answer is None or not str(answer).strip():
            return question, ""

        answer_str = str(answer).strip()
        if not self.shuffle_sentences:
            return question, answer_str

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

class MerlinDataset(CTRATEDataset):
    def _get_vision_input_path(self, image_id, split_path=None) -> Path:
        if self.vision_input_type == VisionInputType.EMBEDDING:
            return Path(self.vision_input_path / f"{image_id}{self.image_id_suffix}_embedding.bl2")
        elif self.vision_input_type == VisionInputType.IMAGE:
            return Path(self.vision_input_path / f"{image_id}.b2nd")
        else:
            raise ValueError(f"Unsupported vision input type: {self.vision_input_type}")

    def shuffle_sentences(self, question: str, answer: str) -> tuple[str, str]:
        """Organ-aware sentence shuffling for CT reports.

        For structured reports (with organ headers like "Liver:", "LUNGS:"):
        - Splits findings into organ sections
        - Shuffles sentences WITHIN each organ section
        - Shuffles the ORDER of organ sections

        For unstructured reports (no organ headers):
        - Shuffles all sentences (each is typically one organ observation)

        For impression:
        - Shuffles lines (each is an independent diagnostic point)
        """

        # print("answer before shuffling:", answer)
        if answer is None or not str(answer).strip():
            return question, ""

        answer_str = str(answer).strip()
        answer_str = re.sub(r"(?<=\.)\s+:", "", answer_str)


        if random.uniform(0, 1) >= self.sentence_shuffle_p:
           return question, answer_str

        # --- Split FINDINGS / IMPRESSION ---
        impression_split = re.split(
            r"(IMPRESSION:)", answer_str, maxsplit=1, flags=re.IGNORECASE
        )

        if len(impression_split) == 3:
            findings_part = impression_split[0].strip()
            impression_header = impression_split[1]
            impression_part = impression_split[2].strip()
        else:
            findings_part = answer_str
            impression_header = None
            impression_part = None

        # Strip optional "FINDINGS:" prefix
        findings_body = re.sub(
            r"^FINDINGS:\s*", "", findings_part, flags=re.IGNORECASE
        )
        had_findings_header = findings_body != findings_part

        # --- Split findings into organ sections ---
        # Organ headers: capitalized phrase followed by ":"
        # Split at boundaries where ". " is followed by an organ header
        organ_header_re = re.compile(
            r"(?<=\.\s)"                          # lookbehind: sentence end ". "
            r"(?=[A-Z][A-Za-z \-/,()]*?:\s)"     # lookahead: organ header
        )

        starts_with_header = bool(
            re.match(r"^[A-Z][A-Za-z \-/,()]*?:\s", findings_body)
        )
        sections_raw = organ_header_re.split(findings_body)
        sections_raw = [s.strip() for s in sections_raw if s.strip()]

        is_structured = len(sections_raw) > 1 or starts_with_header

        if is_structured:
            shuffled_sections = []
            for section in sections_raw:
                # Extract organ header
                header_match = re.match(
                    r"^([A-Z][A-Za-z \-/,()]*?:\s)", section
                )
                if header_match:
                    header = header_match.group(1)
                    body = section[header_match.end() :]
                else:
                    header = ""
                    body = section

                # Shuffle sentences within this organ section
                sentences = [
                    s
                    for s in re.split(r"(?<=[.!?])\s+", body)
                    if s.strip()
                ]
                if len(sentences) > 1:
                    random.shuffle(sentences)

                shuffled_sections.append(header + " ".join(sentences))

            # Shuffle organ section order
            random.shuffle(shuffled_sections)
            shuffled_findings = " ".join(shuffled_sections)
        else:
            # Unstructured: shuffle all sentences
            sentences = [
                s
                for s in re.split(r"(?<=[.!?])\s+", findings_body)
                if s.strip()
            ]
            if len(sentences) > 1:
                random.shuffle(sentences)
            shuffled_findings = " ".join(sentences)

        # Re-add FINDINGS header
        if had_findings_header:
            shuffled_findings = "FINDINGS: " + shuffled_findings

        result = shuffled_findings

        # --- Shuffle impression items (each on its own line after preprocessing) ---
        if impression_header and impression_part:
            impression_sentences = [
                s.strip() for s in impression_part.split("\n") if s.strip()
            ]
            summary_idx = next(
                (
                    idx
                    for idx, sentence in enumerate(impression_sentences)
                    if re.match(r"^SUMMARY\b:?", sentence, flags=re.IGNORECASE)
                ),
                None,
            )
            if summary_idx is None:
                shuffled_impression_sentences = impression_sentences
                summary_sentences = []
            else:
                shuffled_impression_sentences = impression_sentences[:summary_idx]
                summary_sentences = impression_sentences[summary_idx:]

            if len(shuffled_impression_sentences) > 1:
                random.shuffle(shuffled_impression_sentences)

            impression_sentences = shuffled_impression_sentences + summary_sentences
            result += f"\n{impression_header}\n" + "\n".join(impression_sentences)

        # print("answer after shuffling:", result)

        return question, result


class CTRATEDatasetRetrieval(CTRATEDataset):
    def __init__(
        self,
        cfg: RunConfig,
        vision_encoder_patch_size: Tuple[int, ...],
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__(
            cfg=cfg,
            vision_encoder_patch_size=vision_encoder_patch_size,
            split=split,
        )
        # FIXME: not sure how to handle the case where the training samples are limited by `n_train_samples`
        # E.g. a check would be needed that the image IDs of thre retrieval json are all included in the training subset
        self.retrieval_json_path = Path(cfg.data.retrieval_json_path)
        # TODO: provide this part of the cfg
        self.n_retrieval_samples = int(cfg.data.retrieval_n_samples)
        self.maybe_add_vision_tokens = ""
        if self.split in ["val", "test"]:
            self.dataset_json_path_for_retrieval = Path(cfg.data.train_dataset_json_path)
            with open(self.dataset_json_path_for_retrieval, "r") as f:
                self.samples_retrieval = json.load(f)
        elif self.split == "train":
            self.samples_retrieval = self.samples

        self.additional_instruction_prompt = "\nConsider those reports which were retrieved based on the query image."
        self.add_image = False

        with open(self.retrieval_json_path, "r") as f:
            self.retrieval_dict = self._sanatize_retrieval_dict(json.load(f), self.samples_retrieval)

        self._validate_retrieval_dict_covers_samples(self.samples, self.retrieval_dict)

    @staticmethod
    def _sanatize_retrieval_dict(
            retrieval_dict: Dict[str, Dict],
            samples_retrieval: List[Dict]
        ) -> Dict[str, Dict]:
        # 1. Create a Set of valid IDs for O(1) lookup speed
        valid_ids = {sample['image_id'] for sample in samples_retrieval}

        for _, data in retrieval_dict.items():
            neighbors = data.get('k_nearest_neighbor', [])
            scores = data.get('similarity', [])
            assert len(neighbors) == len(scores)

            clean_neighbors = []
            clean_scores = []

            for neighbor_id, score in zip(neighbors, scores):
                if neighbor_id in valid_ids:
                    clean_neighbors.append(neighbor_id)
                    clean_scores.append(score)
            data['k_nearest_neighbor'] = clean_neighbors
            data['similarity'] = clean_scores

        return retrieval_dict

    @staticmethod
    def _validate_retrieval_dict_covers_samples(
            samples: List[Dict],
            retrieval_dict: Dict[str, Dict],
        ) -> None:
        missing_image_ids = []

        for sample in samples:
            if "image_id" not in sample:
                raise KeyError(
                    "Encountered a training sample without an `image_id` key while validating retrieval coverage."
                )

            image_id = sample["image_id"]
            if image_id not in retrieval_dict:
                missing_image_ids.append(image_id)

        if missing_image_ids:
            n_missing = len(missing_image_ids)
            n_samples = len(samples)
            preview_missing = missing_image_ids[:10]
            raise KeyError(
                "Retrieval dictionary is missing entries for training samples. "
                f"Missing {n_missing}/{n_samples} image_ids. "
                f"First missing ids: {preview_missing}"
            )

    @staticmethod
    def get_case_from_dataset_json(dataset_json: List[Dict], image_id: str) -> Dict[str, str]:
        for case in dataset_json:
            if case["image_id"] == image_id:
                return case

        raise KeyError(
            f"Image id: {image_id} can not be found in provided dataset_json make sure it exists"
        )

    def modify_input(self, formatted_qa, formatted_qa_prompt_only, result):
        if self.n_retrieval_samples == 0:
            return formatted_qa, formatted_qa_prompt_only

        query_img_id = result["image_id"]

        # get image_ids of most similar
        nearest_neighbors_ids = self.retrieval_dict[query_img_id]["k_nearest_neighbor"]
        nearest_neighbors_ids = nearest_neighbors_ids[:min(len(nearest_neighbors_ids), self.n_retrieval_samples)]

        # formatted_qa and formatted_qa_prompt_only reference the same "user_message" object
        # we only need to modify one of them here
        assert formatted_qa[1] is formatted_qa_prompt_only[1]
        # for input_prompt in [formatted_qa, formatted_qa_prompt_only]:
        # get user input
        for d in formatted_qa:
            if d["role"] == "user":
                d["content"].append({"type": "text", "text": self.additional_instruction_prompt})
                for n_nn, nearest_neighbor_id in enumerate(nearest_neighbors_ids):
                    case = self.get_case_from_dataset_json(self.samples_retrieval, nearest_neighbor_id)
                    answer = case["answer"]
                    if self.add_image:
                        d["content"].append({"type": "video", "vision_input_path": nearest_neighbor_id})
                    d["content"].append({"type": "text", "text": f"Report {n_nn + 1}:\n{answer}"})

        return formatted_qa, formatted_qa_prompt_only

    # def modify_sample(self, result: Dict) -> Dict:
    #     question = result["question"]
    #     query_img_id = result["image_id"]

    #     # get image_ids of most similar
    #     nearest_neighbors_ids = self.retrieval_dict[query_img_id]["k_nearest_neighbor"]

    #     # Add text to the question/input prompt
    #     question += (
    #         f"\nHere is a list of {self.n_retrieval_samples} similar "
    #         + "CT-reports based on the similarity of the query scan.\n"
    #     )
    #     for n_nn, nearest_neighbor_id in enumerate(nearest_neighbors_ids):
    #         case = self.get_case_from_dataset_json(self.samples, nearest_neighbor_id)
    #         answer = case["answer"]
    #         question += f"Report {n_nn + 1}{self.maybe_add_vision_tokens}:\n{answer}\n"

    #     result["question"] = question
    #     return result


class CTRATEDatasetRetrievalWithImage(CTRATEDatasetRetrieval):
    def __init__(
        self,
        cfg: RunConfig,
        vision_encoder_patch_size: Tuple[int, ...],
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__(
            cfg=cfg,
            vision_encoder_patch_size=vision_encoder_patch_size,
            split=split,
        )
        # self.maybe_add_vision_tokens = "<|vision_start|><|video_pad|><|vision_end|>"
        self.maybe_add_vision_tokens = ""
        self.additional_instruction_prompt = "\nConsider those similar report/image pairs which were retrieved based on the query image."
        self.add_image = True


    # TODO: instead of overriding the signature of the base class refactor base class and introduce hooks where this can be used
    def _get_vision_input_path(self, image_id: str) -> List[Path]:
        # get image_ids of most similar
        nearest_neighbors_ids = self.retrieval_dict[image_id]["k_nearest_neighbor"]
        nearest_neighbors_ids = nearest_neighbors_ids[:min(len(nearest_neighbors_ids), self.n_retrieval_samples)]

        # Bind the super method to a variable outside the list comprehension as else might lead to TypeError in some python versions
        parent_get_vision_input_path = super()._get_vision_input_path

        # TODO: the fix with split path is VERY hacky and not sure if it is correct overall...
        paths = [parent_get_vision_input_path(nn_id, split_path="train") for nn_id in nearest_neighbors_ids]

        paths.insert(0, parent_get_vision_input_path(image_id))  # add query image at first position
        return paths

    def _load_vision_input(self, vision_input_path: List[Path]):
        parent_load_vision_input = super()._load_vision_input
        res = [parent_load_vision_input(p) for p in vision_input_path]
        # FIXME: this function assumes that each input has the same size...
        # assert [len(r.shape) == 5 for r in res], "Each loaded vision input should be 5D (B, C, H, W, D)"
        # return np.concatenate(res, axis=0)  # concatenate along batch dimension
        # Flatten again
        # res = [image_in_case for same_case in res for image_in_case in same_case]
        return res

    # TODO is this the same as in the parent class?
    def _load_embeddings(self, sample, result: Dict) -> Dict:
        vision_input_path = self._get_vision_input_path(sample["image_id"])
        vision_input = self._load_vision_input(vision_input_path)

        if self.embedding_constant is not None:
            if type(self.embedding_constant) is float:
                result["vision_input"] = np.full(
                    vision_input.shape, self.embedding_constant, dtype=np.float32
                )
            elif type(self.embedding_constant) is str and self.embedding_constant == "random":
                result["vision_input"] = np.random.rand(*vision_input.shape).astype(np.float32)
            else:
                raise ValueError(
                    f"Unsupported embedding constant type: {type(self.embedding_constant)}"
                )
        else:
            if self.embedding_transforms is not None:
                vision_input = [
                    self.embedding_transforms(embedding) for embedding in vision_input
                ]
            result["vision_input"] = vision_input

        return result


class CTRATEDatasetRetrievalWithImagePrompt1(CTRATEDatasetRetrievalWithImage):
    def __init__(
        self,
        cfg: RunConfig,
        vision_encoder_patch_size: Tuple[int, ...],
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__(
            cfg=cfg,
            vision_encoder_patch_size=vision_encoder_patch_size,
            split=split,
        )
        # self.maybe_add_vision_tokens = "<|vision_start|><|video_pad|><|vision_end|>"
        self.maybe_add_vision_tokens = ""
        self.additional_instruction_prompt = "\nRetrieved Similar Cases:"
        self.add_image = True

    def format_into_message(
            self, question, answer, system_text=SYSTEM_TEXT
    ):

        assert not self.text_only
        system_text = "You are a medical imaging AI assistant specialized in analyzing CT scans. Your task is to generate a structured radiology report based on the provided CT scan, using retrieved similar cases as contextual guidance."
        system_message = {"role": "system", "content": [{"type": "text", "text": system_text}]}

        question_p1 = """
Carefully analyze the query image and consider the retrieved image-report pairs. Use the retrieved reports to:
- Understand common language and structure for describing findings.
- Identify potential pathologies that may appear in similar scans.
- Avoid contradicting visual evidence in the query image.

However, do **not** copy findings verbatim. Always base your analysis on the **query image**. If the retrieved cases differ significantly, prioritize what you observe.

Output a report in the following format:

**Findings:**
- Describe observed abnormalities (e.g., location, size, density, morphology).
- Mention normal findings when relevant for context.
- Use precise anatomical terminology.

**Impression:**
- Summarize the most significant findings.
- Suggest possible diagnoses or differential considerations if supported by imaging.
- Avoid overinterpretation or speculation beyond visual evidence.

Now analyze the following:
        """

        if "qwen" in self.model_type.lower():
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": question_p1}] + ([{"type": "video"}] if not self.text_only else [])
            }
        else:
            raise ValueError(f"Unsupported model name: {self.model_type}")

        answer_message = {"role": "assistant", "content": [{"type": "text", "text": answer}]}

        full_conversation = [system_message, user_message, answer_message]
        prompt_only_conversation = [system_message, user_message]
        return full_conversation, prompt_only_conversation

    def modify_input(self, formatted_qa, formatted_qa_prompt_only, result):
        if self.n_retrieval_samples == 0:
            return formatted_qa, formatted_qa_prompt_only

        query_img_id = result["image_id"]

        # get image_ids of most similar
        nearest_neighbors_ids = self.retrieval_dict[query_img_id]["k_nearest_neighbor"]
        nearest_neighbors_ids = nearest_neighbors_ids[:min(len(nearest_neighbors_ids), self.n_retrieval_samples)]

        # formatted_qa and formatted_qa_prompt_only reference the same "user_message" object
        # we only need to modify one of them here
        assert formatted_qa[1] is formatted_qa_prompt_only[1]
        # for input_prompt in [formatted_qa, formatted_qa_prompt_only]:
        # get user input
        for d in formatted_qa:
            if d["role"] == "user":
                d["content"].append({"type": "text", "text": self.additional_instruction_prompt})
                for n_nn, nearest_neighbor_id in enumerate(nearest_neighbors_ids):
                    case = self.get_case_from_dataset_json(self.samples_retrieval, nearest_neighbor_id)
                    answer = case["answer"]
                    d["content"].append({"type": "text", "text": f"[Case {n_nn + 1}]\n"})
                    if self.add_image:
                        d["content"].append({"type": "video", "vision_input_path": nearest_neighbor_id})
                    d["content"].append({"type": "text", "text": f"Report:\n{answer}"})

        return formatted_qa, formatted_qa_prompt_only


class BonnDataset(CTRATEDataset):
    def shuffle_sentences(self, question: str, answer: str) -> tuple[str, str]:
        """
        Randomly shuffle sentences within Findings and Impression sections separately
        """
        return question, answer


class MerlinDatasetRetrieval(MerlinDataset, CTRATEDatasetRetrieval):
    pass


class MerlinDatasetRetrievalWithImage(MerlinDataset, CTRATEDatasetRetrievalWithImage):
    # Use Merlin path conventions for both query and retrieved neighbor images.
    def _get_vision_input_path(self, image_id: str, split_path=None) -> List[Path]:
        nearest_neighbors_ids = self.retrieval_dict[image_id]["k_nearest_neighbor"]
        nearest_neighbors_ids = nearest_neighbors_ids[
            : min(len(nearest_neighbors_ids), self.n_retrieval_samples)
        ]

        paths = [MerlinDataset._get_vision_input_path(self, nn_id) for nn_id in nearest_neighbors_ids]
        paths.insert(0, MerlinDataset._get_vision_input_path(self, image_id))
        return paths


class BonnDatasetRetrieval(BonnDataset, CTRATEDatasetRetrieval):
    pass


class BonnDatasetRetrievalWithImage(BonnDataset, CTRATEDatasetRetrievalWithImage):
    pass


class ToothFairy4Dataset(MedicalImageDataset):
    """
    ToothFairy4 dataset - dental CBCT scans with reports.
    """

    def _get_vision_input_path(self, image_id) -> Path:
        if self.vision_input_type == VisionInputType.IMAGE:
            return self.vision_input_path / f"{image_id}.nii.gz"
        elif self.vision_input_type == VisionInputType.EMBEDDING:
            return self.vision_input_path / f"{image_id}{self.image_id_suffix}_embedding.bl2"
        else:
            raise ValueError(f"Unsupported vision input type: {self.vision_input_type}")
