"""Tests for cleaning generated answers before evaluation."""

import pytest

from vlm.utils.decoding import has_thinking_tags, strip_thinking


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Findings: normal.", "Findings: normal."),
        ("<think>\n\n</think>\n\nFindings: normal.", "Findings: normal."),
        ("<think>\nThe lungs look clear.\n</think>\n\nFindings: normal.", "Findings: normal."),
        # Stray tags from a model that answered inside an unclosed think block.
        ("Findings: normal.\n\n</think>\n\nFindings: normal.", "Findings: normal.\n\n\n\nFindings: normal."),
    ],
)
def test_strip_thinking(text, expected):
    assert strip_thinking(text) == expected


def test_has_thinking_tags():
    assert has_thinking_tags("Report.</think>")
    assert has_thinking_tags("<think>")
    assert not has_thinking_tags("Findings: no thinking here.")
