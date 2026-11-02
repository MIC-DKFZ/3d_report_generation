"""AMOS-MM paths and multiple-choice question augmentation."""

import random
import re
from pathlib import Path

from vlm.data.types import VisionInputType
from vlm.utils.regex_patterns import VQA_OPTION_LETTERS_IN_TEXT_PATTERN, VQA_OPTION_PATTERN

from .base import MedicalImageDataset


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
        do_transforms (bool, optional): If True, augment the images of the training split
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
        if random.uniform(0, 1) >= self.sentence_shuffle_p:
            return question, answer

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
