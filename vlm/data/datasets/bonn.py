"""Bonn report dataset."""

from typing import ClassVar

from .ctrate import CTRATEDataset


class BonnDataset(CTRATEDataset):
    dataset_name: ClassVar[str] = "bonn"

    def shuffle_sentences(self, question: str, answer: str) -> tuple[str, str]:
        """
        Randomly shuffle sentences within Findings and Impression sections separately
        """
        return question, answer
