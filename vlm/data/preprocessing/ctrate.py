"""Shared text formatting for CT-RATE report-generation datasets."""

import re


def clean_text(text: str) -> str:
    """Trim text, collapse repeated spaces, and remove spaces before dots/semicolons."""
    text = text.strip()
    text = re.sub(r" {2,}", " ", text)
    return re.sub(r" +([.;])", r"\1", text)


def format_report(findings: str, impression: str | None) -> str:
    """Format findings and omit empty or missing impressions."""
    answer = f"Findings: {clean_text(findings)}"
    if impression is not None:
        impression = clean_text(impression)
        if impression and impression != "Not given.":
            answer += f"\nImpression: {impression}"
    return answer
