import unittest

from vlm.data.preprocessing.ctrate import format_report
from vlm.data.preprocessing.datasets import CTRATEDatasetPreprocessor
from vlm.scripts.datasets.ctrate.ctrate_utils import format_report_v1


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


if __name__ == "__main__":
    unittest.main()
