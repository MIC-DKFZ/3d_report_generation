
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate NLG metrics for CTRate")
    parser.add_argument(
        "-r", "--results_directory", type=str, required=True, help="Results directory path to iterate over"
    )
    parser.add_argument("--juwels_mount", type=str, default=None, help="Path to Juwels mount point. If not set, assumes local results.")
    parser.add_argument("--train_validation", type=str, default="train", choices=["train", "validation"], help="Whether to use train or val reports as ground truth.")
    args = parser.parse_args()
    args.results_directory = Path(args.results_directory)
    if args.juwels_mount:
        args.juwels_mount = Path(args.juwels_mount)
    return args

def main_cli():
    args = parse_args()
    main(args)

def main(args):
    if args.juwels_mount:
        if args.results_directory.is_absolute():
            results_directory = Path(str(args.results_directory)[1:])
        else:
            results_directory = args.results_directory
        iter_dir = args.juwels_mount / results_directory
    else:
        iter_dir = args.results_directory
    eval_paths = []
    for p in iter_dir.rglob("*"):
        if p.is_dir() and p.name == "inference":
            if (p / "inference_results.json").exists() and not (p / "metrics.json").exists():
                if args.juwels_mount:
                    rel_path = "/" / p.relative_to(args.juwels_mount)
                    eval_paths.append(rel_path)
                else:
                    eval_paths.append(p)
    gt_report_csv_path = Path(f"reportgen_evaluation/ctrate/data/radiology_text_reports/{args.train_validation}_reports.csv")
    print("Running evaluation on\n", [str(e) for e in eval_paths])
    for eval_path in eval_paths:
        command = f"bash reportgen_evaluation/ctrate/test.sh -p {eval_path / 'inference_results.json'} -o {eval_path}  -g {gt_report_csv_path}"
        if args.juwels_mount:
            command += " -j"
        os.system(command)

if __name__ == "__main__":
    main_cli()