import argparse
import multiprocessing
import re
from collections import defaultdict
from pathlib import Path
from utils import logger
from evaluation.metrics import BLEUScorer, ROUGELScorer, METEORScorer
from utils.helper import chunkify, add_datetime_tag
from utils.load_and_save_files import load_json, save_json
import numpy as np
import csv

from utils.logging import setup_logger, log_banner
from utils.regex_patterns import VQA_ANSWER_PATTERN


def eval_report_generation(preds: list, gts: list) -> dict[str, list] | None:
    assert len(preds) == len(gts)
    if not preds:
        raise ValueError("VQA EVALUATION: received empty lists")

    results = {}

    bleu_scorer = BLEUScorer()
    rougel_scorer = ROUGELScorer()
    meteor_scorer = METEORScorer()

    bleu_orders = [1, 2, 3, 4]
    try:
        for order in bleu_orders:
            results[f"bleu_score_{order}"] = bleu_scorer(preds=preds, gts=gts, max_order=order)
    except Exception as e:
        logger.error(f"BLEU score calculation failed: {e}")
        for order in bleu_orders:
            results[f"bleu_score_{order}"] = [-1.0]  # use -1 to indicate error

    # === METEOR ===
    try:
        results["meteor_score"] = meteor_scorer(preds=preds, gts=gts)
    except Exception as e:
        logger.error(f"METEOR score calculation failed: {e}")
        results["meteor"] = [-1.0]  # use -1 to indicate error

    # === ROUGE ===
    try:
        results["rougel_score"] = rougel_scorer(preds=preds, gts=gts)
    except Exception as e:
        logger.error(f"ROUGE-L score calculation failed: {e}")
        results["rougel"] = [-1.0]  # use -1 to indicate error
    return results

def eval_vqa(preds: list, gts: list) -> dict[str, list] | None:
    assert len(preds) == len(gts)
    if not preds:
        raise ValueError("VQA EVALUATION: received empty lists")

    results = {}

    answer_preds = []
    answer_gts = []
    matches = []

    for pred, gt in zip(preds, gts):
        m = re.search(VQA_ANSWER_PATTERN, pred)
        if m:
            answer_pred = m.group(2)  # Extract the letter
        else:
            answer_pred = "No answer found"

        m = re.search(VQA_ANSWER_PATTERN, gt)
        if m:
            answer_gt = m.group(2)
        else:
            raise ValueError(f"Could not find answer in groundtruth: {gt}")

        answer_preds.append(answer_pred)
        answer_gts.append(answer_gt)
        matches.append(answer_pred == answer_gt)

    results["answer_pred"] = answer_preds
    results["answer_gt"] = answer_gts
    results["match"] = matches

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision Language Model Evaluation")

    parser.add_argument(
        "--task_type",
        type=str,
        choices=["VQA", "RG"],
        help="Supported task types: Report Generation (RG); Visual Question Answering (VQA)."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a specific checkpoint directory where the inference results are located."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of subprocesses to run evaluation in parallel. If None, use half the number of CPUs in the system."
    )

    args, _ = parser.parse_known_args()
    return args


def main_cli():
    args = parse_args()
    main(args)


def main(args):
    ckpt_dir = Path(args.checkpoint_dir)
    inference_dir = ckpt_dir / "inference"
    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    log_file_name = add_datetime_tag("eval") + ".log"
    log_file_path = Path(inference_dir, log_file_name)
    setup_logger(log_file_path)
    
    task_type = args.task_type

    inference_results_json_path = inference_dir / "inference_results.json"
    inference_results = load_json(inference_results_json_path)
    num_samples = len(inference_results)
    logger.info_main(f"Loaded {num_samples} samples")

    num_processes = args.num_processes
    if args.num_processes is None:
        num_processes = min(num_samples, multiprocessing.cpu_count() // 2)
    logger.info_main(f"Distributing evaluation to {num_processes} subprocesses")

    csv_results = defaultdict(list)
    for dic in inference_results:
        csv_results["image_id"].append(dic["image_id"])
        csv_results["pred"].append(dic["output_text_clean"])
        csv_results["gt"].append(dic["answer"])

    preds, gts = csv_results["pred"], csv_results["gt"]
    pred_chunks = chunkify(preds, num_chunks=num_processes)
    gt_chunks = chunkify(gts, num_chunks=num_processes)

    if task_type == "VQA":
        eval_func = eval_vqa
    elif task_type == "RG":
        eval_func = eval_report_generation
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    with multiprocessing.Pool(processes=num_processes) as pool:
        result_chunks: list[dict[str, list]] = pool.starmap(func=eval_func, iterable=zip(pred_chunks, gt_chunks))

    results: dict[str, list] = defaultdict(list)
    for d in result_chunks:
        for k, lis in d.items():
            results[k].extend(lis)

    # Handle numeric metric calculation failures indicated by -1
    for k, lis in results.items():
        if isinstance(lis[0], float) and -1.0 in lis:
            results[k] = [-1.0] * num_samples
            logger.error_main(f"Detected failure in calculating {k!r}, set all values to -1 for this metric")

    csv_results.update(results)

    csv_path = inference_dir / "VLM_eval_results.csv"
    fieldnames = tuple(csv_results.keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(fieldnames)
        writer.writerows(zip(*(csv_results[field] for field in fieldnames)))
    logger.info_main(f"Saved sample-wise evaluation results to {str(csv_path)}")

    metric_results = {}
    if task_type == "VQA":
        metric_results["accuracy"] = np.mean(results["match"])
    elif task_type == "RG":
        for nlp_metric_name, values in results.items():
            metric_results[nlp_metric_name] = np.mean(values).item()
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    logger.info_main("[METRICS] Evaluation Summary:")
    for metric_name, value in metric_results.items():
        logger.info_main(f"    > {metric_name}: {value:.4f}")

    metrics_path = inference_dir / "VLM_metrics.json"
    save_json(metric_results, metrics_path)
    logger.info_main(f"Saved metric results to {str(metrics_path)}")

    log_banner("Successfully finished Evaluation")


if __name__ == "__main__":
    main_cli()