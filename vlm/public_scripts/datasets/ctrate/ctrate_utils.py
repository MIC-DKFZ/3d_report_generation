"""Shared CT-RATE dataset JSON generation."""

import argparse
import json
import logging
from pathlib import Path
import re

import pandas as pd

RESOURCE_DIR = Path(__file__).resolve().parent


def clean_text(text):
    text = re.sub(r" {2,}", " ", text.strip())
    return re.sub(r" +([.;])", r"\1", text)


def format_report(findings, impression):
    answer = f"Findings: {clean_text(findings)}"
    if impression is not None:
        impression = clean_text(impression)
        if impression and impression != "Not given.":
            answer += f"\nImpression: {impression}"
    return answer


def format_report_v1(findings, impression):
    """Format a report for nnFoundation (v1), keeping "Not given." impressions."""

    def clean(text):
        text = re.sub(r" {2,}", " ", text.strip())
        return text.replace(" . ", ". ")

    answer = f"Findings: {clean(findings)}"
    if impression is not None:
        answer += f"\nImpression: {clean(impression)}"
    return answer


def entries_from_csv(path, excluded_ids, keep_missing_findings, v1_format=False):
    reports = pd.read_csv(path)
    missing = {"VolumeName", "Findings_EN", "Impressions_EN"} - set(reports.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    entries = []
    seen = set()
    formatter = format_report_v1 if v1_format else format_report
    for _, row in reports.iterrows():
        image_id = row["VolumeName"].replace(".nii.gz", "")
        if image_id in excluded_ids:
            continue
        findings = row["Findings_EN"]
        if pd.isna(findings):
            if not keep_missing_findings:
                continue
            findings = ""
        impression = row["Impressions_EN"]
        answer = formatter(findings, None if pd.isna(impression) else impression)
        if image_id in seen:
            raise ValueError(f"Duplicate VolumeName in {path}: {image_id}")
        seen.add(image_id)
        entries.append(
            {
                "image_id": image_id,
                "question": "Generate a detailed report and diagnosis for this CT scan.",
                "answer": answer,
                "full_answer": answer,
                "answer_type": "TEXT",
                "answer_subtype": "REPORT_DIAGNOSIS",
                "system_text": "You are a medical imaging AI trained to analyze CT scans.",
            }
        )
    return sorted(entries, key=lambda item: item["image_id"])


def partition_training(records, split_ids):
    train_ids = set(split_ids["train"])
    val_ids = set(split_ids["val"])
    if train_ids & val_ids:
        raise ValueError("Patient IDs overlap between the train and validation splits")
    splits = {"train": [], "val": []}
    for record in records:
        patient_id = "_".join(record["image_id"].split("_")[:2])
        if patient_id in train_ids:
            splits["train"].append(record)
        elif patient_id in val_ids:
            splits["val"].append(record)
        else:
            raise ValueError(f"Patient ID not found in either split: {patient_id}")
    return splits


def main(version):
    parser = argparse.ArgumentParser(description=f"Create {version} CT-RATE dataset JSONs.")
    parser.add_argument(
        "--dataset-path",
        "--dataset_path",
        type=Path,
        required=True,
        help="Source root containing radiology_text_reports/train_reports.csv and validation_reports.csv.",
    )
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=RESOURCE_DIR / "selection.json",
        help="Bundled version-specific exclusion and patient split IDs.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    reports_dir = args.dataset_path / "radiology_text_reports"
    required_inputs = [
        reports_dir / "train_reports.csv",
        reports_dir / "validation_reports.csv",
        args.selection_file,
    ]
    for path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required source or bundled support file missing: {path}")
    outputs = [
        args.output_dir / f"ctrate_{version}_{split}.json"
        for split in ("train", "val", "test", "train_val")
    ]
    if {p.resolve() for p in outputs} & {p.resolve() for p in required_inputs}:
        raise ValueError("Output files must not overwrite source or bundled support files")
    selection = json.loads(args.selection_file.read_text())
    exclusions = set(selection["exclusions"][version])
    combined = entries_from_csv(
        reports_dir / "train_reports.csv",
        exclusions,
        keep_missing_findings=(version == "v1"),
        v1_format=(version == "v1"),
    )
    test = entries_from_csv(
        reports_dir / "validation_reports.csv",
        exclusions,
        keep_missing_findings=(version == "v1"),
        v1_format=(version == "v1"),
    )
    splits = {
        **partition_training(combined, selection["splits"]),
        "test": test,
        "train_val": combined,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, entries in splits.items():
        path = args.output_dir / f"ctrate_{version}_{split}.json"
        path.write_text(json.dumps(entries, indent=2) + "\n")
        logging.info("Saved %d samples to %s", len(entries), path)
