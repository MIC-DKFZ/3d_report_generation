import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from vlm.data.preprocessing.ctrate import format_report
from vlm.data.preprocessing.datasets import CTRATEDatasetPreprocessor
from vlm.public_scripts.datasets.ctrate.ctrate_utils import format_report_v1
from vlm.scripts.data.ctrate import create_dataset_jsons_v1, create_dataset_jsons_v2
from vlm.scripts.data.ctrate.create_dataset_jsons_v1 import create_train_val


class TestCTRATEJSONCreation(unittest.TestCase):
    def test_v1_preserves_historical_punctuation_and_placeholder(self):
        self.assertEqual(
            format_report_v1("  Clear  lungs . No effusion ; stable .  ", "Not given."),
            "Findings: Clear lungs. No effusion ; stable .\nImpression: Not given.",
        )
        self.assertEqual(format_report_v1("Normal .", None), "Findings: Normal .\nImpression: ")
        self.assertEqual(
            format_report_v1("Mass.", " . Consolidation area . . Nodules ."),
            "Findings: Mass.\nImpression:. Consolidation area.. Nodules .",
        )

    def test_both_generators_use_workspace_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            for subdir in ("radiology_text_reports", "lung_volumes", "exclusions", "splits"):
                (inputs / subdir).mkdir(parents=True)
            train_ids = ["train_1_a_1", "train_2_a_1", "train_3_a_1"]
            for filename, ids in (
                ("train_reports.csv", train_ids),
                ("validation_reports.csv", ["valid_1_a_1", "valid_2_a_1"]),
            ):
                pd.DataFrame(
                    [
                        {
                            "VolumeName": f"{image_id}.nii.gz",
                            "Findings_EN": "Normal .",
                            "Impressions_EN": "Not given.",
                        }
                        for image_id in ids
                    ]
                ).to_csv(inputs / "radiology_text_reports" / filename, index=False)
            for split, ids in (("train", train_ids), ("test", ["valid_1_a_1", "valid_2_a_1"])):
                pd.DataFrame(
                    [
                        {
                            "filepath": f"{image_id}.nii.gz",
                            "volume_ml": (
                                100 if image_id in {"train_2_a_1", "valid_2_a_1"} else 2000
                            ),
                        }
                        for image_id in ids
                    ]
                ).to_csv(
                    inputs / "lung_volumes" / f"lung_volumes_{split}_ctratev2.csv",
                    index=False,
                )
            (inputs / "exclusions/corrupt_scans_ids.txt").write_text("train_9510_a_1\n")
            (inputs / "exclusions/rocket_exclusion_ids.txt").write_text("train_2_a_1\n")
            (inputs / "exclusions/rocket_test_exclusion_ids.txt").write_text("valid_2_a_1\n")
            (inputs / "splits/train_val_split.json").write_text(
                json.dumps({"train": ["train_1", "train_2"], "val": ["train_3"]})
            )
            for generator in (create_dataset_jsons_v2, create_dataset_jsons_v1):
                with patch("sys.argv", ["generate", "--workspace_dir", str(root)]):
                    generator.main()
            for split in ("train", "val", "test"):
                rows = json.loads((root / f"outputs/ctrate_v2_{split}.json").read_text())
                self.assertEqual(len(rows), 1)
                self.assertFalse((root / f"ctrate_{split}.json").exists())
            self.assertTrue((root / "intermediate/v2/raw_ctrate_v2_train.json").is_file())
            self.assertEqual(
                (root / "intermediate/v2/exclusions/brain_scans_train_ids.txt").read_text(),
                "train_2_a_1\n",
            )
            rocket = json.loads((root / "outputs/ctrate_v1_train_val.json").read_text())
            self.assertEqual([r["image_id"] for r in rocket], ["train_1_a_1", "train_3_a_1"])
            rocket_test = json.loads((root / "outputs/ctrate_v1_test.json").read_text())
            self.assertEqual([r["image_id"] for r in rocket_test], ["valid_1_a_1"])
            self.assertEqual(
                {path.name for path in (root / "outputs").iterdir()},
                {
                    f"ctrate_{version}_{split}.json"
                    for version in ("v1", "v2")
                    for split in ("train", "val", "test", "train_val")
                },
            )
            for version in ("v1", "v2"):
                splits = {
                    split: json.loads(
                        (root / f"outputs/ctrate_{version}_{split}.json").read_text()
                    )
                    for split in ("train", "val", "train_val")
                }
                self.assertEqual(
                    sorted(splits["train"] + splits["val"], key=lambda row: row["image_id"]),
                    splits["train_val"],
                )

    def test_normal_pipeline_still_drops_missing_findings(self):
        row = {"VolumeName": "train_4677_a_1.nii.gz", "Findings_EN": float("nan")}
        self.assertIsNone(CTRATEDatasetPreprocessor._load_csv_row((row, "train", None)))

    def test_missing_impressions_and_whitespace(self):
        for impression in (None, "", "  ", "Not given."):
            with self.subTest(impression=impression):
                self.assertEqual(
                    format_report("  Normal   lungs .  ", impression), "Findings: Normal lungs."
                )
        self.assertEqual(
            format_report("A  nodule .", "Follow-up ; advised ."),
            "Findings: A nodule.\nImpression: Follow-up; advised.",
        )

    def test_generates_from_csv_with_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports_dir = root / "radiology_text_reports"
            reports_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "VolumeName": "train_4677_a_2.nii.gz",
                        "Findings_EN": "Normal .",
                        "Impressions_EN": "Not given.",
                    },
                    {
                        "VolumeName": "train_10011_a_1.nii.gz",
                        "Findings_EN": "A  nodule .",
                        "Impressions_EN": "Follow-up ; advised .",
                    },
                    {
                        "VolumeName": "train_9510_a_1.nii.gz",
                        "Findings_EN": "Exclude this.",
                        "Impressions_EN": "Not given.",
                    },
                    {
                        "VolumeName": "train_4677_a_1.nii.gz",
                        "Findings_EN": "",
                        "Impressions_EN": "Emphysema .",
                    },
                ]
            ).to_csv(reports_dir / "train_reports.csv", index=False)
            exclusion_txt = root / "exclusions.txt"
            exclusion_txt.write_text("train_9510_a_1\n")
            output = root / "output.json"
            records = create_train_val(root, exclusion_txt, output)
            self.assertEqual(
                [r["image_id"] for r in records],
                ["train_10011_a_1", "train_4677_a_1", "train_4677_a_2"],
            )
            self.assertEqual(json.loads(output.read_text()), records)
            self.assertEqual(
                records[0]["answer"], "Findings: A nodule .\nImpression: Follow-up ; advised ."
            )
            self.assertEqual(records[1]["answer"], "Findings: \nImpression: Emphysema .")
            self.assertEqual(records[2]["answer"], "Findings: Normal .\nImpression: Not given.")
            self.assertTrue(all(r["answer"] == r["full_answer"] for r in records))
            self.assertTrue(all(r["answer_subtype"] == "REPORT_DIAGNOSIS" for r in records))
            with self.assertRaises(FileNotFoundError):
                create_train_val(root, root / "missing.txt", output)
            with self.assertRaisesRegex(ValueError, "different from the input"):
                create_train_val(root, exclusion_txt, exclusion_txt)


if __name__ == "__main__":
    unittest.main()
