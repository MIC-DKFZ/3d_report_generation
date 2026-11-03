"""Command-line entry point for evaluating saved VLM inference results."""

import argparse
from pathlib import Path

import pandas as pd

from vlm.evaluation.report_generation import lexical_metrics
from vlm.evaluation.report_generation.class_list import evaluate_class_lists
from vlm.registries.dataset_registry import get_class_list_names
from vlm.registries.evaluation_registry import get_classifier_evaluator
from vlm.utils import logger
from vlm.utils.class_list import strip_class_list
from vlm.utils.serialization import load_json, load_yaml, save_json
from vlm.utils.logging import log_banner, setup_logger
from vlm.utils.run_naming import add_datetime_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision Language Model Evaluation")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the dataset; defaults to the one in the inference directory's final_config.yml.",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default=None,
        help="Path to a specific directory where the inference results are located.",
    )
    parser.add_argument(
        "--gt_labels_path",
        type=Path,
        default=None,
        help="Optional reference label CSV; overrides Merlin's gt_labels.csv.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of subprocesses to run evaluation in parallel. "
        "If None, use half the number of CPUs in the system.",
    )

    args, _ = parser.parse_known_args()
    return args


def main_cli():
    args = parse_args()
    main(args)


def dataset_name_of_inference(inference_dir: Path) -> str:
    """Read the dataset name from the config that inference saved with its results."""
    config_path = inference_dir / "final_config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found; pass --dataset_name.")
    return load_yaml(config_path)["data"]["dataset_name"]


def main(args):
    """Evaluate reports with Pycocoeval and the dataset-specific classifier."""

    inference_dir = Path(args.inference_dir)
    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")
    dataset_name = getattr(args, "dataset_name", None) or dataset_name_of_inference(inference_dir)

    log_file_name = add_datetime_tag("eval") + ".log"
    log_file_path = Path(inference_dir, log_file_name)
    setup_logger(log_file_path)

    inference_results_json_path = inference_dir / "inference_results.json"
    inference_results = load_json(inference_results_json_path)
    num_samples = len(inference_results)
    logger.info_main(f"Loaded {num_samples} samples")

    image_ids, raw_preds, raw_gts = [], [], []
    for dic in inference_results:
        image_ids.append(dic["image_id"])
        raw_preds.append(dic["output_text_clean"])
        raw_gts.append(dic["answer"])

    # Score only the report text; class lists are scored separately below.
    # Remove '\n' here as this will interfere Pycocoeval (specifically METEOR)
    preds = [strip_class_list(pred).replace("\n", " ") for pred in raw_preds]
    gts = [strip_class_list(gt).replace("\n", " ") for gt in raw_gts]

    aggregate_results = {}
    nlp_aggregate, nlp_per_case = lexical_metrics.evaluate_reports(preds, gts)
    aggregate_results.update(nlp_aggregate)

    radbert_metrics = {}
    radbert_pred_df = radbert_gt_df = None
    evaluate_classifier = get_classifier_evaluator(dataset_name)
    if evaluate_classifier is not None:
        classifier_options = {}
        if getattr(args, "gt_labels_path", None) is not None:
            classifier_options["gt_labels_path"] = args.gt_labels_path
        radbert_metrics, radbert_pred_df, radbert_gt_df = evaluate_classifier(
            preds, gts, image_ids, **classifier_options
        )

    aggregate_results.update(radbert_metrics)

    class_list_metrics, class_list_pred_df = evaluate_class_lists(
        raw_preds, raw_gts, get_class_list_names(dataset_name), radbert_gt_df
    )
    aggregate_results.update({f"class_list_{key}": value for key, value in class_list_metrics.items()})
    logger.info_main("[METRICS] Evaluation Summary:")
    for metric_name, value in aggregate_results.items():
        logger.info_main(f"    > {metric_name}: {value:.4f}")

    metrics_path = inference_dir / "VLM_metrics.json"
    save_json(aggregate_results, metrics_path)
    logger.info_main(f"Saved metric results to {str(metrics_path)}")

    # --- per-case NLP metrics ---
    per_case_df = pd.DataFrame({"image_id": image_ids, "pred": preds, "gt": gts})
    for metric_name, scores in nlp_per_case.items():
        per_case_df[metric_name] = scores
    per_case_path = inference_dir / "metrics_per_case.csv"
    per_case_df.to_csv(per_case_path, index=False)
    logger.info_main(f"Saved per-case NLP metrics to {str(per_case_path)}")

    # --- RadBERT class predictions ---
    if radbert_pred_df is not None:
        radbert_out = pd.DataFrame({"image_id": image_ids})
        for col in radbert_pred_df.columns:
            radbert_out[f"pred_{col}"] = radbert_pred_df[col].values
        for col in radbert_gt_df.columns:
            radbert_out[f"gt_{col}"] = radbert_gt_df[col].values
        radbert_path = inference_dir / "radbert_predictions.csv"
        radbert_out.to_csv(radbert_path, index=False)
        logger.info_main(f"Saved RadBERT class predictions to {str(radbert_path)}")

    # --- Class list predictions ---
    if class_list_pred_df is not None:
        class_list_out = pd.DataFrame({"image_id": image_ids})
        for col in class_list_pred_df.columns:
            class_list_out[f"pred_{col}"] = class_list_pred_df[col].values
        class_list_path = inference_dir / "class_list_predictions.csv"
        class_list_out.to_csv(class_list_path, index=False)
        logger.info_main(f"Saved class list predictions to {str(class_list_path)}")

    log_banner("Successfully finished Evaluation")


if __name__ == "__main__":
    main_cli()
