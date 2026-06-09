import argparse
import multiprocessing
import os
import random
import re
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from torch.utils.data import SequentialSampler, DataLoader

from vlm.evaluation.ctrate_radbert.calculate_metrics import calculate_metrics
from vlm.evaluation.ctrate_radbert.utils import CTDataset, RadBertClassifier, RadBertInference, ctrate_label_cols
from vlm.utils import logger
from vlm.evaluation.hf_evaluate.scorers import BLEUScorer, ROUGELScorer, METEORScorer
from vlm.utils.helper import add_datetime_tag, chunkify
from vlm.utils.load_and_save_files import load_json, save_json
from vlm.utils.logging import setup_logger, log_banner
from vlm.utils.regex_patterns import VQA_ANSWER_PATTERN


def eval_report_generation_pycocoeval(preds: list, gts: list) -> tuple[dict, dict]:
    """
    Returns (aggregate_results, per_case_results).
    per_case_results maps metric names to lists of per-sample scores (same order as preds/gts).
    Failures are indicated by -1.0 in per_case_results and -1.0 in aggregate_results.
    """
    assert len(preds) == len(gts)
    if not preds:
        raise ValueError("RG EVALUATION: received empty lists")

    n = len(preds)
    aggregate = {}
    per_case = {}

    bleu_scorer = Bleu(4)
    meteor_scorer = Meteor()
    rougel_scorer = Rouge()
    cider_scorer = Cider()

    # Put preds and gts into compatible format for Pycocoeval
    in_res = {i: [pred] for i, pred in enumerate(preds)}
    in_gts = {i: [gt] for i, gt in enumerate(gts)}

    # === BLEU ===
    try:
        (bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4), bleu_per_image = bleu_scorer.compute_score(
            gts=in_gts, res=in_res, verbose=0
        )
        # bleu_per_image: list of 4 arrays, one per n-gram order
        for order, (agg, per) in enumerate(
            zip(
                [bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4],
                bleu_per_image,
            ),
            start=1,
        ):
            aggregate[f"bleu_score_{order}"] = agg
            per_case[f"bleu_score_{order}"] = list(per)
    except Exception as e:
        logger.error(f"BLEU score calculation failed: {e}")
        for order in range(1, 5):
            aggregate[f"bleu_score_{order}"] = -1.0
            per_case[f"bleu_score_{order}"] = [-1.0] * n

    # === METEOR ===
    try:
        meteor_score, meteor_per_image = meteor_scorer.compute_score(gts=in_gts, res=in_res)
        aggregate["meteor_score"] = meteor_score
        per_case["meteor_score"] = list(meteor_per_image)
    except Exception as e:
        logger.error(f"METEOR score calculation failed: {e}")
        meteor_scorer.lock.release()
        aggregate["meteor_score"] = -1.0
        per_case["meteor_score"] = [-1.0] * n

    # === ROUGE ===
    try:
        rougel_score, rougel_per_image = rougel_scorer.compute_score(gts=in_gts, res=in_res)
        aggregate["rougel_score"] = rougel_score.item()
        per_case["rougel_score"] = list(rougel_per_image)
    except Exception as e:
        logger.error(f"ROUGE-L score calculation failed: {e}")
        aggregate["rougel_score"] = -1.0
        per_case["rougel_score"] = [-1.0] * n

    # === CIDEr ===
    try:
        cider_score, cider_per_image = cider_scorer.compute_score(gts=in_gts, res=in_res)
        aggregate["cider_score"] = cider_score.item()
        per_case["cider_score"] = list(cider_per_image)
    except Exception as e:
        logger.error(f"CIDEr score calculation failed: {e}")
        aggregate["cider_score"] = -1.0
        per_case["cider_score"] = [-1.0] * n

    return aggregate, per_case


def inference_ctrate_radbert(reports: list[str], ckpt_path: str | Path, hf_radbert_dir: str | Path):
    df = pd.DataFrame({"reports": reports})
    for col in ctrate_label_cols:
        df[col] = 0

    batch_size = 32
    if os.getenv("SLURM_JOB_ID") or os.getenv("LSB_JOBID"):
        batch_size = 96

    # DataLoader
    ds = CTDataset(df, len(ctrate_label_cols), ctrate_label_cols, hf_radbert_dir)
    loader = DataLoader(
        ds, sampler=SequentialSampler(ds), batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # Model
    model = RadBertClassifier(n_classes=len(ctrate_label_cols), hf_model_dir=hf_radbert_dir)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model.embeddings.position_ids" in state_dict:
        del state_dict["model.embeddings.position_ids"]

    model.load_state_dict(state_dict)
    model = model.cuda().eval()

    trainer = RadBertInference(
        model,
        {"test": loader},
        len(ctrate_label_cols),
        ctrate_label_cols,
    )

    preds = trainer.infer()  # ndarray [N, C]
    result = pd.DataFrame(preds.astype(int), columns=ctrate_label_cols)
    return result


def eval_ctrate_radbert(preds: list, gts: list) -> tuple[dict, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Returns (aggregate_metrics, pred_df, gt_df).
    pred_df / gt_df are [N, C] DataFrames of binary class predictions; None when RadBERT is unavailable.
    """
    radbert_dir = os.getenv("VLM_CTRATE_RADBERT_DIR")
    if radbert_dir is None:
        logger.warning_main("The environment variable 'VLM_CTRATE_RADBERT_DIR' is not set. "
                            "Cannot proceed with RadBERT inference for CT-RATE.")
        return {}, None, None

    radbert_dir = Path(radbert_dir)
    radbert_ckpt = radbert_dir / "RadBertClassifier.pth"
    hf_radbert_dir = radbert_dir / "RadBERT-RoBERTa-4m"

    pred_df = inference_ctrate_radbert(preds, radbert_ckpt, hf_radbert_dir)
    gt_df = inference_ctrate_radbert(gts, radbert_ckpt, hf_radbert_dir)

    results = calculate_metrics(pred_df, gt_df)
    return results, pred_df, gt_df


def eval_report_generation(preds: list, gts: list) -> dict[str, list] | None:
    assert len(preds) == len(gts)
    if not preds:
        raise ValueError("RG EVALUATION: received empty lists")

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


def compare_report_subset(preds: list, output_dir: str | Path, num_pairs: int = 500) -> None:
    """
    Calculate pairwise BLEU-4 scores for `num_pairs` prediction pairs to assess
    whether the model is producing overly similar (template-like) reports.
    """
    if len(preds) < num_pairs * 2:
        num_pairs = len(preds) // 2

    logger.info_main(f"Calculating the BLEU-4 score for {num_pairs} prediction pairs...")

    bleu_scorer = BLEUScorer()

    indices = list(range(len(preds)))
    random.seed(42)
    subset_indices = random.sample(indices, num_pairs*2)

    preds_subset = []
    for idx in subset_indices:
        preds_subset.append(preds[idx])

    try:
        scores = bleu_scorer(preds=preds_subset[:num_pairs], gts=preds_subset[num_pairs:], max_order=4)
        mean_score = np.mean(scores)
    except Exception as e:
        logger.error(f"BLEU score calculation failed: {e}")
        logger.error(f"Skipping subset comparison.")
        return

    logger.info_main(f"Mean BLEU-4 score of {num_pairs} prediction pairs: {mean_score:.4f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f"BLEU-4 Score Distribution of {num_pairs} Prediction Pairs\nMean Score = {mean_score:.4f}")

    ax.hist(scores, bins=30, edgecolor='black')

    ax.set_xlabel("BLEU-4 Score")
    ax.set_ylabel("Frequency")

    output_path = Path(output_dir) / "bleu_score_histogram.png"
    plt.savefig(output_path, bbox_inches='tight')

    plt.close(fig)
    logger.info_main(f"Histogram saved to {output_path}")


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
        "--dataset_name",
        type=str,
        default="ctrate",
        help="Name of the dataset."
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["VQA", "RG"],
        help="Supported task types: Report Generation (RG); Visual Question Answering (VQA)."
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default=None,
        help="Path to a specific directory where the inference results are located."
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


def main_(args):
    """
    Use this main function for HF evaluate or VQA
    """
    inference_dir = Path(args.inference_dir)
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
        compare_report_subset(preds, inference_dir, num_pairs=500)
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


def main(args):
    """
    Use this main function for pycocoeval
    """

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
    nlp_aggregate, nlp_per_case = eval_report_generation_pycocoeval(preds, gts)
    aggregate_results.update(nlp_aggregate)

    radbert_pred_df = radbert_gt_df = None
    if args.dataset_name == "ctrate":
        radbert_metrics, radbert_pred_df, radbert_gt_df = eval_ctrate_radbert(preds, gts)
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