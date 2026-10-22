"""Shared MERLIN dataset JSON generation."""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
RESOURCE_DIR = Path(__file__).resolve().parent


def clean_text_v2(text):
    text = text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" +([.;])", r"\1", text)
    return text


def entry_from_report(row, report_text):
    """Build a dataset entry without modifying its report text."""
    return {
        "image_id": row["study id"],
        "question": "Generate a detailed report and diagnosis for this CT scan.",
        "answer": report_text,
        "full_answer": report_text,
        "answer_type": "TEXT",
        "answer_subtype": "REPORT_DIAGNOSIS",
        "system_text": "You are a medical imaging AI trained to analyze CT scans.",
    }


def standardize_entry_v2(row):
    """Format a report using the v2 section and whitespace rules."""
    findings_text = clean_text_v2(str(row["Findings"]))

    # Normalize FINDINGS header to uppercase
    findings_text = re.sub(r"^findings:", "FINDINGS:", findings_text, flags=re.IGNORECASE)

    # Normalize IMPRESSION header to uppercase (if present)
    findings_text = re.sub(r"\bimpression\s*:", "IMPRESSION:", findings_text, flags=re.IGNORECASE)
    findings_text = re.sub(r"(^|\n)[ \t]*IMPRESSION:", r"\1IMPRESSION:", findings_text)
    findings_text = re.sub(r"([^\n])[ \t]*IMPRESSION:", r"\1\nIMPRESSION:", findings_text)

    # If no IMPRESSION header but numbered items exist after findings, insert it
    if "IMPRESSION:" not in findings_text:
        # Look for the first numbered item pattern (e.g. "\n1. ")
        match = re.search(r"\n\d+\.\s+", findings_text)
        if match:
            insert_pos = match.start()
            findings_text = (
                findings_text[:insert_pos] + "\nIMPRESSION:\n" + findings_text[insert_pos + 1 :]
            )

    # Remove enumeration numbers in IMPRESSION (e.g., "1. ", "2. ")
    # Only remove numbers that appear after the IMPRESSION header
    impression_match = re.search(r"IMPRESSION:", findings_text)
    if impression_match:
        before = findings_text[: impression_match.end()]
        after = findings_text[impression_match.end() :]
        after = after.lstrip(" \t\n")
        after = re.sub(r"(?:^|\n)\d+\.\s+", "\n", after)
        after = after.strip("\n")
        findings_text = before + "\n" + after

    return entry_from_report(row, findings_text)


def load_reports(reports_xlsx: Path) -> pd.DataFrame:
    if not reports_xlsx.is_file():
        raise FileNotFoundError(reports_xlsx)
    df = pd.read_excel(reports_xlsx, keep_default_na=False)
    missing_columns = {"study id", "Findings", "Split"} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required report columns: {sorted(missing_columns)}")
    logger.info("Loaded %s reports from %s", len(df), reports_xlsx)
    return df


def get_source_splits(df):
    """Validate the source split labels before selecting records."""
    normalized_splits = df["Split"].astype(str).str.strip().str.lower()
    unexpected_splits = sorted(set(normalized_splits) - {"train", "val", "test"})
    if unexpected_splits:
        raise ValueError(f"Unexpected Split values: {unexpected_splits}")

    return {split: df.loc[normalized_splits == split] for split in ("train", "val", "test")}


def save_dataset_json(split_df, out_path: Path, entry_formatter=standardize_entry_v2):
    """Write sorted records using the requested report formatter."""
    entries = [entry_formatter(row) for _, row in split_df.iterrows()]
    entries.sort(key=lambda entry: entry["image_id"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(entries, f, indent=2)
    logger.info("Saved %s entries to %s", len(entries), out_path)
    return entries


def clean_text_v1(text: str) -> str:
    """Apply the v1 whitespace rules."""
    text = re.sub(r" {2,}", " ", text.strip())
    return re.sub(r" +([.;])", r"\1", text)


def standardize_entry_v1(row):
    """Format a report for nnFoundation (v1), keeping impression headers and numbering."""
    report = re.sub(r"^FINDINGS:\s*", "", str(row["Findings"]).strip())
    return entry_from_report(row, clean_text_v1(report))


def frozen_splits(df, split_ids):
    source = get_source_splits(df)
    ids = split_ids["val"]
    if not isinstance(ids, list) or any(
        not isinstance(image_id, str) or not image_id.strip() for image_id in ids
    ):
        raise ValueError("splits.val must be a JSON list of nonempty study IDs")
    if not ids:
        raise ValueError("Validation ID list is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate validation identifiers in splits.val")
    ids = set(ids)
    val_candidates = source["val"]
    missing = ids - set(val_candidates["study id"])
    if missing:
        raise ValueError(
            f"{len(missing)} bundled validation IDs are absent from the filtered "
            f"source validation split: {sorted(missing)[:10]}"
        )
    selected = val_candidates["study id"].isin(ids)
    return {
        "train": pd.concat([source["train"], val_candidates.loc[~selected]], ignore_index=True),
        "val": val_candidates.loc[selected],
        "test": source["test"],
    }


def main(version):
    parser = argparse.ArgumentParser(description=f"Create {version} MERLIN dataset JSONs.")
    parser.add_argument("--reports-xlsx", "--reports_xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=RESOURCE_DIR / "selection.json",
        help="Bundled version-specific exclusion and study split IDs.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    selection = json.loads(args.selection_file.read_text())
    df = load_reports(args.reports_xlsx)
    df = df.loc[~df["study id"].isin(selection["exclusions"][version])]
    if df["study id"].duplicated().any():
        raise ValueError("Duplicate study IDs remain; add them to the selection file exclusions")
    splits = frozen_splits(df, selection["splits"])
    combined = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    formatter = standardize_entry_v1 if version == "v1" else standardize_entry_v2
    for split, rows in {**splits, "train_val": combined}.items():
        save_dataset_json(rows, args.output_dir / f"merlin_{version}_{split}.json", formatter)
