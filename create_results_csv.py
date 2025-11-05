import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import yaml

METRICS_DICT = {
    "CRG Score": ("metrics.json", ["crg", "CRG"]),
    "F1": ("metrics.json", ["classification", "macro", "f1"]),
    "Accuracy": ("metrics.json", ["classification", "macro", "accuracy"]),
    "Precision": ("metrics.json", ["classification", "macro", "precision"]),
    "Recall": ("metrics.json", ["classification", "macro", "recall"]),
    "CIDEr": ("metrics.json", ["generation", "CIDEr"]),
    "ROUGE-L": ("metrics.json", ["generation", "ROUGE_L"]),
    "ROUGE-L (HF)": ("VLM_metrics.json", ["rougel_score"]),
    "BLEU-1": ("metrics.json", ["generation", "BLEU_1"]),
    "BLEU-1 (HF)": ("VLM_metrics.json", ["bleu_score_1"]),
    "BLEU-2": ("metrics.json", ["generation", "BLEU_2"]),
    "BLEU-2 (HF)": ("VLM_metrics.json", ["bleu_score_2"]),
    "BLEU-3": ("metrics.json", ["generation", "BLEU_3"]),
    "BLEU-3 (HF)": ("VLM_metrics.json", ["bleu_score_3"]),
    "BLEU-4": ("metrics.json", ["generation", "BLEU_4"]),
    "BLEU-4 (HF)": ("VLM_metrics.json", ["bleu_score_4"]),
    "BLEU (mean)": ("metrics.json", ["generation", "BLEU_mean"]),
    # "BLEU (mean) (HF)": ("VLM_metrics.json", ["generation", "bleu_score_mean"]),
    "METEOR": ("metrics.json", ["generation", "METEOR"]),
    "METEOR (HF)": ("VLM_metrics.json", ["meteor_score"]),
}

def get_config_value(config, key):
    keys = key.split(".")
    value = config
    for k in keys:
        value = value.get(k, None)
    return value

def get_metrics_value(metrics_path: Path, key):
    metrics_file, metrics_keys = METRICS_DICT[key]
    with open(metrics_path / metrics_file, "r") as f:
        metrics = json.load(f)
    value = metrics
    for mk in metrics_keys:
        value = value.get(mk, None)
    return value

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate NLG metrics for CTRate")
    parser.add_argument(
        "-r", "--results_directory", type=str, required=True, help="Results directory path to iterate over"
    )
    parser.add_argument("--config_keys", type=str, nargs="+", default=[], help="Configuration keys to include in the CSV.")
    args = parser.parse_args()
    args.results_directory = Path(args.results_directory)
    return args

def main_cli():
    args = parse_args()
    main(args)

def main(args):
    iter_dir = args.results_directory
    eval_paths = {}
    for p in iter_dir.rglob("*"):
        if p.is_dir() and p.name == "inference":
            if (p / "metrics.json").exists() and (p / "VLM_metrics.json").exists():
                print(p)
                final_config_path = p.parent.parent / "final_config.yml"
                eval_paths[p] = final_config_path
    metrics_list = ["CRG Score", "F1", "Accuracy", "Precision", "Recall", "CIDEr", "ROUGE-L", "ROUGE-L (HF)", "BLEU-1", "BLEU-1 (HF)", "BLEU-2", "BLEU-2 (HF)", "BLEU-3", "BLEU-3 (HF)", "BLEU-4", "BLEU-4 (HF)", "BLEU (mean)", "BLEU (mean) (HF)", "METEOR", "METEOR (HF)"]
    results_df = pd.DataFrame(columns=[*args.config_keys, *metrics_list])
    for inference_path, config_path in eval_paths.items():
        result = {}
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        for key in args.config_keys:
            result[key] = get_config_value(config, f"run.{key}")
        for metric in metrics_list:
            if metric != "BLEU (mean) (HF)":
                result[metric] = get_metrics_value(inference_path, metric)
        result["BLEU (mean) (HF)"] = (result["BLEU-1 (HF)"] + result["BLEU-2 (HF)"] + result["BLEU-3 (HF)"] + result["BLEU-4 (HF)"]) / 4
        result = pd.Series(result)
        results_df = pd.concat([results_df, result.to_frame().T], ignore_index=True)
    results_df.to_csv(iter_dir / "aggregated_results.csv", index=False)


if __name__ == "__main__":
    main_cli()