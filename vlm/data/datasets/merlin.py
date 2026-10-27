"""Merlin paths and structured report augmentation."""

import random
import re
from typing import ClassVar

from .ctrate import CTRATEDataset


class MerlinDataset(CTRATEDataset):
    dataset_name: ClassVar[str] = "merlin"

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
