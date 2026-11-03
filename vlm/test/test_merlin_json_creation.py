"""Report formatting of the v1 Merlin dataset JSONs, against the v2 one."""

from vlm.scripts.datasets.merlin.merlin_utils import (
    standardize_entry_v1,
    standardize_entry_v2,
)


def test_v1_keeps_inline_header_and_leading_numbering():
    row = {
        "study id": "case",
        "Findings": "  Findings: normal   lungs . impression: 1. Unchanged .\n2. Follow-up ;  ",
    }
    expected = "Findings: normal lungs. impression: 1. Unchanged.\n2. Follow-up;"
    entry = standardize_entry_v1(row)
    assert entry["answer"] == entry["full_answer"] == expected
    assert standardize_entry_v2(row)["answer"] != expected


def test_v1_retains_carriage_returns():
    row = {
        "study id": "case",
        "Findings": (
            "  FINDINGS: normal   lungs .\r\nImpression: 1. Unchanged .\r\n2. Follow-up ;  "
        ),
    }
    expected = "normal lungs.\r\nImpression: 1. Unchanged.\r\n2. Follow-up;"
    entry = standardize_entry_v1(row)
    assert entry["answer"] == entry["full_answer"] == expected


def test_v1_preserves_report_numbers_and_existing_headers():
    row = {
        "study id": "case",
        "Findings": "FINDINGS: Lesion in segment\n3. Stable.\nIMPRESSION:\n1. Follow-up.",
    }
    expected = "Lesion in segment\n3. Stable.\nIMPRESSION:\n1. Follow-up."
    entry = standardize_entry_v1(row)
    assert entry["answer"] == entry["full_answer"] == expected
