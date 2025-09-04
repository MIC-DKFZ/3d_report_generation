import argparse

import os
from pathlib import Path
from typing import Union


def main(results_path: Path, gt_report_csv_path: Path, output_dir: Union[Path, None] = None):
    if output_dir is None:
        print(f"Output directory not specified. Using default: {results_path.parent}")
        output_dir = results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    os.system(
        f"bash reportgen_evaluation/test.sh -o {output_dir} -p {results_path} -g {gt_report_csv_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Evaluation")
    parser.add_argument(
        "--results_path", type=Path, required=True, help="Directory of inference results"
    )
    parser.add_argument(
        "--gt_report_csv_path", type=Path, required=True, help="Path to ground truth report CSV"
    )
    parser.add_argument("--output_dir", type=Path, help="Directory to save evaluation outputs")
    args = parser.parse_args()

    main(
        results_path=args.results_path,
        gt_report_csv_path=args.gt_report_csv_path,
        output_dir=args.output_dir,
    )
