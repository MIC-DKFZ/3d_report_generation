"""Bonn report dataset."""

from .ctrate import CTRATEDataset


class BonnDataset(CTRATEDataset):
    def shuffle_sentences(self, question: str, answer: str) -> tuple[str, str]:
        """
        Randomly shuffle sentences within Findings and Impression sections separately
        """
        return question, answer
