"""Generate validation reports and evaluate dataset-specific classifier metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import TrainerCallback
import wandb

from vlm.data.batches import extract_batch_prompt_only
from vlm.utils import logger
from vlm.utils.ddp import gather_inference_outputs
from vlm.utils.decoding import decode_generated_answers, decode_label_answers
from vlm.utils.serialization import save_json

# Lexical metrics are inactive; restore these imports and scorer initialization
# together with the commented calculations below when enabling them.
# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor


class ReportEvaluationCallback(TrainerCallback):
    """Score generated reports and add metrics before checkpoint selection.

    The callback owns a generation dataloader separate from the Trainer's loss
    evaluation dataloader. Classifier errors propagate and stop training.
    """

    def __init__(
        self,
        eval_dataloader: DataLoader,
        max_new_tokens: int,
        dataset_name: str,
        tokenizer,
        *,
        print_first_sample: bool = True,
        is_distributed: bool = False,
        calculate_classifier_score: bool = True,
    ):
        self.eval_dataloader = eval_dataloader
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self.print_first_sample = print_first_sample
        self.is_distributed = is_distributed
        self.classifier_scorer = None
        self.classifier_gt_df = None

        if calculate_classifier_score:
            from vlm.registries.evaluation_registry import get_classifier_evaluator

            self.classifier_scorer = get_classifier_evaluator(dataset_name)
            if self.classifier_scorer is None:
                raise ValueError(
                    f"No classifier evaluator registered for dataset {dataset_name!r}."
                )

        # self.bleu_scorer = Bleu(4)
        # self.meteor_scorer = Meteor()
        # self.rougel_scorer = Rouge()
        # self.cider_scorer = Cider()

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        logger.info_main(f"Starting report evaluation at epoch {state.epoch}.")
        was_training = model.training
        model.eval()
        try:
            results = self._generate_reports(model, args.device)
            if self.is_distributed:
                results = gather_inference_outputs(local_results=results, main_rank=0)

            metric_results = {}
            if state.is_world_process_zero:
                if not results["image_ids"]:
                    raise ValueError("Report evaluation received no validation samples.")
                logger.info_main(f"Evaluating {len(results['image_ids'])} generated reports.")
                metric_results = self._compute_metrics(results, args.output_dir, state.global_step)

            if self.is_distributed:
                payload = [metric_results]
                dist.broadcast_object_list(payload, src=0)
                metric_results = payload[0]

            if metrics is not None:
                metrics.update(metric_results)
            # Trainer.evaluate logs its metrics before this hook, so extend that logged entry.
            last_log = state.log_history[-1] if state.log_history else {}
            if last_log.get("step") == state.global_step and "eval_loss" in last_log:
                last_log.update(metric_results)
        finally:
            model.train(was_training)

        logger.info_main("Finished report evaluation.")

    def _generate_reports(self, model, device) -> dict[str, list]:
        results = {"preds": [], "gts": [], "image_ids": []}
        with torch.inference_mode():
            for inputs in self.eval_dataloader:
                results["image_ids"].extend(inputs["raw_samples"]["image_id"])
                prompt_inputs = extract_batch_prompt_only(inputs).to(device)
                del prompt_inputs["raw_samples"]
                del prompt_inputs["batch_count"]
                prompt_input_ids = prompt_inputs["input_ids"]
                outputs = model.generate(**prompt_inputs, max_new_tokens=self.max_new_tokens)
                results["gts"].extend(decode_label_answers(self.tokenizer, inputs["labels"]))
                results["preds"].extend(
                    decode_generated_answers(self.tokenizer, outputs, prompt_input_ids)
                )
                del prompt_inputs, outputs
                torch.cuda.empty_cache()

        if self.print_first_sample and results["preds"]:
            logger.info_main(f"First sample GROUNDTRUTH:\n{results['gts'][0]}")
            logger.info_main(f"First sample PREDICTION:\n{results['preds'][0]}")

        # Remove newlines for the inactive Pycocoeval calculations (especially METEOR).
        results["preds"] = [pred.replace("\n", " ") for pred in results["preds"]]
        results["gts"] = [gt.replace("\n", " ") for gt in results["gts"]]
        return results

    def _compute_metrics(self, results: dict[str, list], output_dir, global_step: int) -> dict:
        all_preds = results["preds"]
        all_gts = results["gts"]
        all_image_ids = results["image_ids"]

        # Put preds and gts into compatible format for Pycocoeval
        # in_res = {i: [pred] for i, pred in enumerate(all_preds)}
        # in_gts = {i: [gt] for i, gt in enumerate(all_gts)}

        # === BLEU ===
        # try:
        #     (bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4), _ = self.bleu_scorer.compute_score(gts=in_gts,
        #                                                                                                  res=in_res,
        #                                                                                                  verbose=0)
        # except Exception as e:
        #     logger.error(f"BLEU score calculation failed: {e}")
        #     bleu_score_1 = bleu_score_2 = bleu_score_3 = bleu_score_4 = -1.0

        # === METEOR ===
        # try:
        #     meteor_score, _ = self.meteor_scorer.compute_score(gts=in_gts, res=in_res)
        # except Exception as e:
        #     logger.error(f"METEOR score calculation failed: {e}")
        #     self.meteor_scorer.lock.release()
        #     meteor_score = -1.0  # use -1 to indicate error

        # === ROUGE ===
        # try:
        #     rougel_score, _ = self.rougel_scorer.compute_score(gts=in_gts, res=in_res)
        #     rougel_score = rougel_score.item()
        # except Exception as e:
        #     logger.error(f"ROUGE-L score calculation failed: {e}")
        #     rougel_score = -1.0  # use -1 to indicate error

        # === CIDEr ===
        # try:
        #     cider_score, _ = self.cider_scorer.compute_score(gts=in_gts, res=in_res)
        #     cider_score = cider_score.item()
        # except Exception as e:
        #     logger.error(f"CIDEr score calculation failed: {e}")
        #     cider_score = -1.0  # use -1 to indicate error

        metric_results = {
            # "bleu_score_1": bleu_score_1,
            # "bleu_score_2": bleu_score_2,
            # "bleu_score_3": bleu_score_3,
            # "bleu_score_4": bleu_score_4,
            # "meteor_score": meteor_score,
            # "rougel_score": rougel_score,
            # "cider_score": cider_score,
        }
        classifier_scores = {}
        pred_df = gt_df = None
        if self.classifier_scorer is not None:
            classifier_scores, pred_df, gt_df = self.classifier_scorer(
                preds=all_preds,
                gts=all_gts,
                image_ids=all_image_ids,
                default_gt_df=self.classifier_gt_df,
            )
            if "f1" not in classifier_scores:
                raise ValueError("Classifier evaluation did not return the required 'f1' metric.")
            if self.classifier_gt_df is None:
                self.classifier_gt_df = gt_df

        callback_eval_dir = Path(output_dir) / "callback_eval" / f"step-{global_step}"
        callback_eval_dir.mkdir(parents=True, exist_ok=True)
        if self.classifier_scorer is not None:
            for key, value in classifier_scores.items():
                logger.info_main(f"classifier_{key}: {value:.4f}")
            if wandb.run is not None:
                wandb.log({f"classifier_{key}": value for key, value in classifier_scores.items()})
            if pred_df is not None:
                self._save_classifier_predictions(callback_eval_dir, all_image_ids, pred_df, gt_df)

        # Keep unprefixed scores on disk so earlier callback evaluations remain plottable.
        metrics_path = callback_eval_dir / "VLM_metrics.json"
        save_json({**metric_results, **classifier_scores}, metrics_path)
        logger.info_main(f"Saved metric results to {metrics_path}")
        if self.classifier_scorer is not None:
            self.save_f1_plot(callback_eval_dir.parent)

        return {
            **{f"eval_{key}": value for key, value in metric_results.items()},
            **{f"eval_classifier_{key}": value for key, value in classifier_scores.items()},
        }

    @staticmethod
    def _save_classifier_predictions(output_dir, image_ids, pred_df, gt_df):
        predictions = pd.DataFrame({"image_id": image_ids})
        for col in pred_df.columns:
            predictions[f"pred_{col}"] = pred_df[col].values
        for col in gt_df.columns:
            predictions[f"gt_{col}"] = gt_df[col].values
        output_path = output_dir / "classifier_predictions.csv"
        predictions.to_csv(output_path, index=False)
        logger.info_main(f"Saved classifier predictions to {output_path}")

    @staticmethod
    def _step_from_callback_eval_dir(step_dir: Path) -> int:
        prefix = "step-"
        if not step_dir.name.startswith(prefix):
            raise ValueError(f"Expected step directory to start with '{prefix}': {step_dir}")
        return int(step_dir.name.removeprefix(prefix))

    def load_f1_scores(self, callback_eval_root: Path) -> list[tuple[int, float]]:
        scores: list[tuple[int, float]] = []
        for metrics_path in callback_eval_root.glob("step-*/VLM_metrics.json"):
            with metrics_path.open() as f:
                metrics = json.load(f)

            f1 = metrics.get("f1")
            if f1 is None:
                continue

            scores.append((self._step_from_callback_eval_dir(metrics_path.parent), float(f1)))

        return sorted(scores)

    def save_f1_plot(self, callback_eval_root: Path) -> None:
        scores = self.load_f1_scores(callback_eval_root)
        if not scores:
            logger.warning_main(
                f"No F1 scores found in callback eval metrics under {callback_eval_root}"
            )
            return

        steps = [step for step, _ in scores]
        f1_scores = [f1 for _, f1 in scores]
        best_step, best_f1 = max(scores, key=lambda item: item[1])

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(
            steps,
            f1_scores,
            marker="o",
            linewidth=2,
            markersize=4,
        )
        ax.scatter([best_step], [best_f1], s=60, zorder=3)
        ax.set_title(f"Callback Evaluation F1 | best {best_f1:.4f} at step {best_step}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("F1")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        output_path = callback_eval_root / "f1_score.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        logger.info_main(f"Saved callback F1 plot to {str(output_path)}")
