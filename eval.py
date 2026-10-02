"""Command-line entry point for evaluating saved VLM inference results."""

import argparse
from pathlib import Path

import pandas as pd

from vlm.evaluation.report_generation import lexical_metrics
from vlm.registries.evaluation_registry import get_classifier_evaluator
from vlm.utils import logger
from vlm.utils.load_and_save_files import load_json, save_json
from vlm.utils.logging import log_banner, setup_logger
from vlm.utils.run_naming import add_datetime_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision Language Model Evaluation")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ctrate",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default=None,
        help="Path to a specific directory where the inference results are located.",
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


def main(args):
    """Evaluate reports with Pycocoeval and the dataset-specific classifier."""

    inference_dir = Path(args.inference_dir)
    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    log_file_name = add_datetime_tag("eval") + ".log"
    log_file_path = Path(inference_dir, log_file_name)
    setup_logger(log_file_path)

    inference_results_json_path = inference_dir / "inference_results.json"
    inference_results = load_json(inference_results_json_path)
    num_samples = len(inference_results)
    logger.info_main(f"Loaded {num_samples} samples")

    image_ids, preds, gts = [], [], []
    for dic in inference_results:
        image_ids.append(dic["image_id"])
        preds.append(dic["output_text_clean"])
        gts.append(dic["answer"])

    # Remove '\n' here as this will interfere Pycocoeval (specifically METEOR)
    preds = [pred.replace("\n", " ") for pred in preds]
    gts = [gt.replace("\n", " ") for gt in gts]

    aggregate_results = {}
    nlp_aggregate, nlp_per_case = lexical_metrics.evaluate_reports(preds, gts)
    aggregate_results.update(nlp_aggregate)

    radbert_metrics = {}
    radbert_pred_df = radbert_gt_df = None
    evaluate_classifier = get_classifier_evaluator(args.dataset_name)
    if evaluate_classifier is not None:
        radbert_metrics, radbert_pred_df, radbert_gt_df = evaluate_classifier(
            preds, gts, image_ids
        )

    aggregate_results.update(radbert_metrics)
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

    log_banner("Successfully finished Evaluation")


if __name__ == "__main__":
    main_cli()
