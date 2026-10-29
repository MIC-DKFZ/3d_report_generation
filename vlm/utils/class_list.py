"""Abnormality class lists appended to reports as an '<answer>…</answer>' block."""

import re

CLASS_LIST_BLOCK_PATTERN = re.compile(r"\s*<answer>(.*?)</answer>", flags=re.DOTALL)
TRAILING_CLASS_LIST_PATTERN = re.compile(r"\s*<answer>.*?</answer>\s*\Z", flags=re.DOTALL)


def split_class_list(text: str) -> tuple[str, str]:
    """Split a trailing class list block, with its leading whitespace, from the report."""
    match = TRAILING_CLASS_LIST_PATTERN.search(text)
    if match is None:
        return text, ""
    return text[: match.start()], text[match.start():]


def has_class_list(text: str) -> bool:
    return CLASS_LIST_BLOCK_PATTERN.search(text) is not None


def strip_class_list(text: str) -> str:
    """Remove every class list block, e.g. to score only the report text."""
    return CLASS_LIST_BLOCK_PATTERN.sub("", text).strip()


def class_name_pattern(name: str) -> re.Pattern:
    """Match a class name as whole words, e.g. 'Pleural effusion' in 'bilateral pleural-effusions'."""
    words = r"[\s_-]+".join(re.escape(word) for word in name.split())
    return re.compile(rf"\b{words}(?:s|es|e)?\b", flags=re.IGNORECASE)


def parse_class_list(text: str, class_names) -> list[int]:
    """
    Mark which of 'class_names' the last class list block of 'text' contains.

    Entries are comma-separated; a class counts when some entry contains its name (see
    'class_name_pattern'). Other entries are ignored, and a missing block counts as no classes.
    """
    blocks = CLASS_LIST_BLOCK_PATTERN.findall(text)
    entries = [entry.strip() for entry in blocks[-1].split(",")] if blocks else []
    return [
        int(any(class_name_pattern(name).search(entry) for entry in entries))
        for name in class_names
    ]
