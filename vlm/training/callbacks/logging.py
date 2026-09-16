import math
import os
import re
import json
from pathlib import Path
from typing import Literal
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import wandb
from transformers.modeling_utils import unwrap_model
from vlm.evaluation.hf_evaluate.scorers import BLEUScorer, ROUGELScorer, METEORScorer
from vlm.utils.decoding import decode_generated_answers, decode_label_answers
from vlm.utils.ddp_utils import gather_inference_outputs
from vlm.utils.filter_batch import extract_batch_prompt_only
from vlm.utils import logger
from vlm.utils.helper import get_trainable_state_dict, is_main_process
from vlm.utils.load_and_save_files import save_json
from vlm.utils.regex_patterns import VQA_ANSWER_PATTERN

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


class PrintEvalBatchCallback(TrainerCallback):
    def __init__(
        self, eval_dataloader: DataLoader, task_type: Literal["VQA", "RG"], max_new_tokens: int,
            tokenizer=None, processor=None, eval_every_n_epochs=0.25, print_first_sample=True, log_n_per_callback=5,
            is_distributed=False
    ):
        self.eval_dataloader = eval_dataloader
        self.task_type = task_type
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer

        self.num_samples = len(eval_dataloader.sampler)  # type: ignore[arg-type]
        self.total_size = len(eval_dataloader.dataset)  # type: ignore[arg-type]

        self.last_epoch = 0
        self.eval_every_n_epochs = eval_every_n_epochs

        self.print_first_sample = print_first_sample
        self.log_n_per_callback = min(self.total_size, log_n_per_callback)
        self.is_distributed = is_distributed

        # Only create wandb table if wandb is initialized (i.e., on main process)
        # if wandb.run is not None and is_main_process():
        #     columns = [
        #         "Epoch",
        #         "Sample Index",
        #         "Generated Output",
        #         "Ground Truth"
        #     ]
        #     if self.task_type == "VQA":
        #         columns += [
        #             "Answer Pred",
        #             "Answer GT"
        #         ]
        #     self.eval_table = wandb.Table(columns=columns, log_mode="MUTABLE")
        #     logger.info_main(f"Created an eval table with the following columns: {columns}")

        self.processor = processor

        # Track best metrics for model saving
        self.best_rougel_score = -1.0    # RG
        self.best_accuracy = -1.0       # VQA

        # RG Metric Scorers
        self.bleu_scorer = BLEUScorer()
        self.rougel_scorer = ROUGELScorer()
        self.meteor_scorer = METEORScorer()

        self.bleu_orders = [1, 2, 3, 4]

    def on_evaluate(
        self,
        args,
        state,
        control,
        model=None,
        eval_dataloader=None,
        metrics=None,
        **kwargs,
    ):
        if state.epoch - self.last_epoch < self.eval_every_n_epochs:
            return

        self.last_epoch = state.epoch

        logger.info_main(f"Starting Evaluation Callback...")
        logger.info_main(f"Epoch {state.epoch} reached")

        # Store local results; if using DDP, all processes will merge into 'all_results' on the main rank.
        results = {}

        # === Run inference ===
        preds = []
        gts = []
        model.eval()  # Ensure model is in evaluation mode
        with torch.inference_mode():  # Disable gradient calculations
            for inputs in self.eval_dataloader:
                inputs_prompt_only = extract_batch_prompt_only(inputs)
                inputs_prompt_only = inputs_prompt_only.to(args.device)
                del inputs_prompt_only["raw_samples"]
                del inputs_prompt_only["batch_count"]

                outputs = model.generate(
                    **inputs_prompt_only,
                    max_new_tokens=self.max_new_tokens
                )

                # Decode input and generated output
                decoded_gts = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
                decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Explicitly delete large tensors to free memory
                del inputs_prompt_only
                del outputs
                torch.cuda.empty_cache()  # Clear CUDA cache periodically

                decoded_gt_assistant_answers = [decoded_gt.split("assistant\n")[-1].strip() for decoded_gt in decoded_gts]
                decoded_pred_assistant_answers = [decoded_output.split("assistant\n")[-1].strip() for decoded_output in decoded_preds]

                # store predictions and ground truths
                preds.extend(decoded_pred_assistant_answers)
                gts.extend(decoded_gt_assistant_answers)

        if self.print_first_sample:
            logger.error_main(f"First sample GROUNDTRUTH:\n{decoded_gt_assistant_answers[0]}")
            logger.error_main(f"First sample PREDICTION:\n{decoded_pred_assistant_answers[0]}")

        # Synchronize all distributed processes to keep logging output aligned
        if self.is_distributed:
            dist.barrier()

        # === Calculate pairwise scores/metrics on local validation subset ===
        if self.task_type == "VQA":
            # === Count correct QA ===
            answer_preds = []
            answer_gts = []
            n_correct_qa = 0

            for pred, gt in zip(preds, gts):
                m = re.search(VQA_ANSWER_PATTERN, pred)
                if m:
                    answer_pred = m.group(2)    # Extract the letter
                else:
                    answer_pred = "No answer found"

                m = re.search(VQA_ANSWER_PATTERN, gt)
                if m:
                    answer_gt = m.group(2)
                else:
                    raise ValueError(f"Could not find answer in groundtruth: {gt}")

                if answer_pred == answer_gt:
                    n_correct_qa += 1

                answer_preds.append(answer_pred)
                answer_gts.append(answer_gt)

            logger.info(f"Local Correct QA: {n_correct_qa}/{self.num_samples}")  # log on every rank

            results["answer_preds"] = answer_preds
            results["answer_gts"] = answer_gts
            results["n_correct_qa"] = [n_correct_qa]

        elif self.task_type == "RG":
            # === BLEU ===
            try:
                for order in self.bleu_orders:
                    results[f"bleu_score_{order}"] = self.bleu_scorer(preds=preds, gts=gts, max_order=order)
            except Exception as e:
                logger.error(f"BLEU score calculation failed: {e}")
                for order in self.bleu_orders:
                    results[f"bleu_score_{order}"] = [-1.0]  # use -1 to indicate error

            # === METEOR ===
            try:
                results["meteor_score"] = self.meteor_scorer(preds=preds, gts=gts)
            except Exception as e:
                logger.error(f"METEOR score calculation failed: {e}")
                results["meteor"] = [-1.0]  # use -1 to indicate error

            # === ROUGE ===
            try:
                results["rougel_score"] = self.rougel_scorer(preds=preds, gts=gts)
            except Exception as e:
                logger.error(f"ROUGE-L score calculation failed: {e}")
                results["rougel"] = [-1.0]  # use -1 to indicate error
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        results["preds"] = preds
        results["gts"] = gts

        # === Gather results from other ranks at main rank process (DDP) ===
        if self.is_distributed:
            all_results = gather_inference_outputs(local_results=results, main_rank=0)
            logger.info_main("Successfully gathered inference results at main rank.")
            if is_main_process():
                _lengths = {k: len(lis) for k, lis in all_results.items()}
                logger.info_main(f"Lengths: {_lengths}")
        else:
            all_results = results

        # === Calculate final scores/metrics and log to WandB ===
        if state.is_world_process_zero:

            # Handle numeric metric calculation failures indicated by -1
            for k, lis in all_results.items():
                if isinstance(lis[0], float) and -1.0 in lis:
                    all_results[k] = [-1.0] * self.total_size
                    logger.error(f"Detected failure in calculating {k!r}, set all values to -1 for this metric")

            if self.task_type == "VQA":
                n_correct_qa = sum(all_results["n_correct_qa"])
                accuracy = n_correct_qa / self.total_size

                logger.info_main(f"Total Correct QA: {n_correct_qa}/{self.total_size}")
                wandb.log({
                    "correct_qa": n_correct_qa,
                    "accuracy": accuracy
                })

                # === Log and optionally save best model ===
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    logger.info_main(f"New best ACCURACY: {self.best_accuracy:.4f} - Saving model checkpoint")
                    self.save_checkpoint(model, args.output_dir)

                # Always log best, even if there is no improvement
                wandb.log({"best_accuracy": self.best_accuracy})
            elif self.task_type == "RG":
                # Log the BLEU scores
                for order in self.bleu_orders:
                    bleu_score = np.mean(all_results[f"bleu_score_{order}"]).item()
                    logger.info_main(f"bleu_score_{order}: {bleu_score:.4f}")
                    wandb.log({f"bleu_score_{order}": bleu_score})

                # Log the METEOR score
                meteor_score = np.mean(all_results["meteor_score"]).item()
                logger.info_main(f"METEOR score: {meteor_score}")
                wandb.log({"meteor_score": meteor_score})

                # Log the ROUGE score
                rougel_score = np.mean(all_results["rougel_score"]).item()
                logger.info_main(f"ROUGE-L score: {rougel_score}")
                wandb.log({"rougel_score": rougel_score})

                # === Log and optionally save best model ===
                if rougel_score > self.best_rougel_score:
                    self.best_rougel_score = rougel_score
                    logger.info_main(f"New best ROUGE-L score: {self.best_rougel_score:.4f} - Saving model checkpoint")
                    self.save_checkpoint(model, args.output_dir)

                # Always log best, even if there is no improvement
                wandb.log({"best_rouge_score": self.best_rougel_score})
            else:
                raise ValueError(f"Unsupported task_type: {self.task_type}")

            # === Log 'self.log_n_per_callback' number of cases to the WandB eval table ===
            # for idx in range(0, self.total_size, self.total_size // self.log_n_per_callback):
            #     table_data = [
            #         state.epoch,
            #         idx,
            #         all_results["preds"][idx],
            #         all_results["gts"][idx]
            #     ]
            #     if self.task_type == "VQA":
            #         table_data += [
            #             all_results["answer_preds"][idx],
            #             all_results["answer_gts"][idx]
            #         ]
            #     self.eval_table.add_data(*table_data)
            #
            # wandb.log({"eval_table": self.eval_table})

        # Ensure that main process finishes writing before continuing
        if self.is_distributed:
            dist.barrier()

        model.train()  # Ensure model is back in training mode

        logger.info_main(f"Finished Evaluation Callback...")


    def save_checkpoint(self, model, output_dir):
        best_model_dir = os.path.join(output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)

        # Save only the trainable state_dict
        model_to_save = unwrap_model(model)
        trainable_state_dict = get_trainable_state_dict(model_to_save)

        # Save the model
        model.save_pretrained(best_model_dir, state_dict=trainable_state_dict)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(best_model_dir)
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(best_model_dir)
        logger.info_main(f"Model saved to: {best_model_dir}")


class PycocoevalCallback(TrainerCallback):
    def __init__(
        self, eval_dataloader: DataLoader, max_new_tokens: int,
            tokenizer=None, processor=None, eval_every_n_epochs=0.5, print_first_sample=True,
            is_distributed=False, calculate_radbert_score=True, dataset_name="ctrate"
    ):
        self.eval_dataloader = eval_dataloader
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer

        # self.num_samples = len(eval_dataloader.sampler)  # type: ignore[arg-type]
        # self.total_size = len(eval_dataloader.dataset)  # type: ignore[arg-type]

        self.last_epoch = 0
        self.eval_every_n_epochs = eval_every_n_epochs

        self.print_first_sample = print_first_sample
        self.is_distributed = is_distributed

        self.processor = processor

        # RG Metric Scorers
        self.bleu_scorer = Bleu(4)
        self.meteor_scorer = Meteor()
        self.rougel_scorer = Rouge()
        self.cider_scorer = Cider()

        self.calculate_radbert_score = calculate_radbert_score
        if self.calculate_radbert_score:
            # Pick the report classifier matching the dataset. Both scorers share
            # the same (preds, gts) -> metrics-dict contract, so everything below
            # (wandb keys, error fallback) is unchanged.
            if dataset_name == "ctrate":
                from eval import eval_ctrate_radbert
                self.radbert_scorer = eval_ctrate_radbert
            elif dataset_name == "merlin":
                from eval import eval_merlin_modernbert
                self.radbert_scorer = eval_merlin_modernbert
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

        self.radbert_gt_df = None


    def on_evaluate(
        self,
        args,
        state,
        control,
        model=None,
        eval_dataloader=None,
        metrics=None,
        **kwargs,
    ):
        self.last_epoch = state.epoch

        logger.info_main(f"Starting Evaluation Callback...")
        logger.info_main(f"Epoch {state.epoch} reached")

        # Store local results; if using DDP, all processes will merge into 'all_results' on the main rank.
        results = {}

        # === Run inference ===
        preds = []
        gts = []
        image_ids = []
        model.eval()  # Ensure model is in evaluation mode
        with torch.inference_mode():  # Disable gradient calculations
            for inputs in self.eval_dataloader:
                image_ids.extend(inputs["raw_samples"]["image_id"])
                inputs_prompt_only = extract_batch_prompt_only(inputs)
                inputs_prompt_only = inputs_prompt_only.to(args.device)
                del inputs_prompt_only["raw_samples"]
                del inputs_prompt_only["batch_count"]

                prompt_input_ids = inputs_prompt_only["input_ids"]
                outputs = model.generate(
                    **inputs_prompt_only,
                    max_new_tokens=self.max_new_tokens
                )

                decoded_gt_assistant_answers = decode_label_answers(
                    self.tokenizer,
                    inputs["labels"],
                )
                decoded_pred_assistant_answers = decode_generated_answers(
                    self.tokenizer,
                    outputs,
                    prompt_input_ids,
                )

                # Explicitly delete large tensors to free memory
                del inputs_prompt_only
                del outputs
                torch.cuda.empty_cache()  # Clear CUDA cache periodically

                # store predictions and ground truths
                preds.extend(decoded_pred_assistant_answers)
                gts.extend(decoded_gt_assistant_answers)

        if self.print_first_sample:
            logger.error_main(f"First sample GROUNDTRUTH:\n{decoded_gt_assistant_answers[0]}")
            logger.error_main(f"First sample PREDICTION:\n{decoded_pred_assistant_answers[0]}")

        # Remove '\n' here as this will interfere Pycocoeval (specifically METEOR)
        results["preds"] = [pred.replace("\n", " ") for pred in preds]
        results["gts"] = [gt.replace("\n", " ") for gt in gts]
        results["image_ids"] = image_ids

        # === Gather results from other ranks at main rank process (DDP) ===
        if self.is_distributed:
            all_results = gather_inference_outputs(local_results=results, main_rank=0)
            logger.info_main("Successfully gathered inference results at main rank.")
            if is_main_process():
                _lengths = {k: len(lis) for k, lis in all_results.items()}
                logger.info_main(f"Lengths: {_lengths}")
        else:
            all_results = results

        # === Calculate final scores/metrics and log to WandB ===
        metric_results = {}
        if state.is_world_process_zero:

            all_preds = all_results["preds"]
            all_gts = all_results["gts"]
            all_image_ids = all_results["image_ids"]

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

            # === RadBert ===
            if self.calculate_radbert_score:
                try:
                    radbert_scores, pred_df, gt_df = self.radbert_scorer(
                        preds=all_preds,
                        gts=all_gts,
                        image_ids=all_image_ids,
                        default_gt_df=self.radbert_gt_df
                    )
                    if self.radbert_gt_df is None:
                        self.radbert_gt_df = gt_df
                except Exception as e:
                    logger.error(f"radbert score calculation failed: {e}")
                    radbert_scores = {
                        "accuracy": -1.0,
                        "balanced_accuracy": -1.0,
                        "precision": -1.0,
                        "recall": -1.0,
                        "f1": -1.0,
                        "crg": -1.0
                    }  # use -1 to indicate error
                    pred_df = gt_df = None

            metric_results = {
                # "bleu_score_1": bleu_score_1,
                # "bleu_score_2": bleu_score_2,
                # "bleu_score_3": bleu_score_3,
                # "bleu_score_4": bleu_score_4,
                # "meteor_score": meteor_score,
                # "rougel_score": rougel_score,
                # "cider_score": cider_score,
            }
            if self.calculate_radbert_score:
                metric_results.update(radbert_scores)

            # Log the BLEU scores
            # logger.info_main(f"BLEU-1 score: {bleu_score_1:.4f}")
            # logger.info_main(f"BLEU-2 score: {bleu_score_2:.4f}")
            # logger.info_main(f"BLEU-3 score: {bleu_score_3:.4f}")
            # logger.info_main(f"BLEU-4 score: {bleu_score_4:.4f}")
            # wandb.log({f"bleu_score_1": bleu_score_1})
            # wandb.log({f"bleu_score_2": bleu_score_2})
            # wandb.log({f"bleu_score_3": bleu_score_3})
            # wandb.log({f"bleu_score_4": bleu_score_4})

            # Log the METEOR score
            # logger.info_main(f"METEOR score: {meteor_score}")
            # wandb.log({"meteor_score": meteor_score})

            # Log the ROUGE-L score
            # logger.info_main(f"ROUGE-L score: {rougel_score}")
            # wandb.log({"rougel_score": rougel_score})

            # Log the CIDEr score
            # logger.info_main(f"CIDEr score: {cider_score}")
            # wandb.log({"cider_score": cider_score})

            callback_eval_dir = (
                Path(args.output_dir) / "callback_eval" / f"step-{state.global_step}"
            )
            callback_eval_dir.mkdir(parents=True, exist_ok=True)

            # Log RadBert
            if self.calculate_radbert_score:
                for key, value in radbert_scores.items():
                    logger.info_main(f"radbert_{key}: {value:.4f}")
                    wandb.log({f"radbert_{key}": value})

                if pred_df is not None:
                    radbert_out = pd.DataFrame({"image_id": all_image_ids})
                    for col in pred_df.columns:
                        radbert_out[f"pred_{col}"] = pred_df[col].values
                    for col in gt_df.columns:
                        radbert_out[f"gt_{col}"] = gt_df[col].values

                    radbert_path = callback_eval_dir / "radbert_predictions.csv"
                    radbert_out.to_csv(radbert_path, index=False)
                    logger.info_main(f"Saved RadBERT class predictions to {str(radbert_path)}")

            metrics_path = callback_eval_dir / "VLM_metrics.json"
            save_json(metric_results, metrics_path)
            logger.info_main(f"Saved metric results to {str(metrics_path)}")
            if self.calculate_radbert_score:
                self.save_f1_plot(callback_eval_dir.parent)

        if self.is_distributed:
            metric_results_list = [metric_results]
            dist.broadcast_object_list(metric_results_list, src=0)
            metric_results = metric_results_list[0]

        if metrics is not None:
            for key, value in metric_results.items():
                metrics_key = (
                    f"eval_radbert_{key}"
                    if key in {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "crg"}
                    else f"eval_{key}"
                )
                metrics[metrics_key] = value

        # Ensure that main process finishes writing before continuing
        if self.is_distributed:
            dist.barrier()

        model.train()  # Ensure model is back in training mode

        logger.info_main(f"Finished Evaluation Callback...")

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

class LossLoggingCallback(TrainerCallback):
    """A custom callback that logs the training loss"""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """
        Event called by Trainer when logging metrics.
        """
        if state.is_world_process_zero and logs is not None:
            # Check if 'loss' is in the logs dictionary (it should be during training)
            if "loss" in logs:
                loss = logs["loss"]
                global_step = state.global_step
                # print(f"Step: {global_step} | Training Loss: {loss:.4f}")
                logger.log("LOSS", f"Step: {global_step} | Training Loss: {loss:.5f}")

            # You could also check for 'eval_loss' here if you wanted to log validation loss
            elif 'eval_loss' in logs:
                eval_loss = logs['eval_loss']
                global_step = state.global_step
                logger.log("LOSS", f"Step: {global_step} | Validation Loss: {eval_loss:.5f}")

            # --- Add your custom logging logic here ---
            # Example: Append to a custom file
            # with open("custom_loss_log.txt", "a") as f:
            #     if 'loss' in logs:
            #         f.write(f"{global_step},{logs['loss']}\n")


class WandbGroupedLRCallback(TrainerCallback):
    """
    Logs learning rates of different parameter groups.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if state.is_world_process_zero and wandb.run is not None:
            optimizer: Optimizer = kwargs["optimizer"]

            named_lrs = {}
            for param_group in optimizer.param_groups:
                module_name = param_group["name"].removesuffix("_DECAY").removesuffix("_NO")
                # It's ok if the 'NO_DECAY' overwrites the 'DECAY' learning rate of the same module or vice versa
                # because they have the same learning rate
                named_lrs[module_name] = param_group["lr"]

            learning_rates = {f"lr/{module_name}": loss for module_name, loss in named_lrs.items()}

            # Log custom metric(s) to WandB
            wandb.log({**learning_rates, "train/global_step": state.global_step})


class TerminateOnNaNLossCallback(TrainerCallback):
    def __init__(self, threshold: int = 3, filename: str = "TRAINING_FAILED.txt"):
        self.threshold = threshold
        self.filename = filename
        self.bad_value_counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        current_loss = logs.get("loss")
        if current_loss is None:
            return

        is_bad_value = (
                math.isnan(current_loss) or
                math.isinf(current_loss) or
                current_loss == 0.0
        )

        if is_bad_value:
            self.bad_value_counter += 1
        else:
            self.bad_value_counter = 0

        if self.bad_value_counter >= self.threshold:
            file_path = Path(args.output_dir, self.filename)
            try:
                with open(file_path, "w") as f:
                    f.write("Status: ABORTED\n")
                    f.write(f"Reason: Loss instability detected for {self.threshold} consecutive logs.\n")
                    f.write(f"Final Value: {current_loss}\n")
                    f.write(f"Global Step: {state.global_step}\n")

                logger.error_main(f"\nABORTING: Crash report written to: {file_path}")
            except Exception as e:
                logger.error_main(f"ABORTING: Could not write crash report: {e}")

            raise RuntimeError(
                f"Training aborted! Loss was {current_loss} for {self.threshold} consecutive logs. "
                f"See {file_path} for details."
            )


class EveryNEpochsCallback(TrainerCallback):
    # `trainer.eval_steps` is passed as `n` and controls the cadence. This callback is
    # needed, for example, to evaluate every n epochs instead of after every epoch.
    def __init__(self, n=1):
        self.n = n

    def _get_epoch(self, state):
        return int(round(state.epoch)) if state.epoch is not None else None

    def _is_final_epoch(self, args, epoch: int | None) -> bool:
        return epoch is not None and epoch >= int(round(args.num_train_epochs))

    def _should_run_on_epoch(self, args, state) -> bool:
        epoch = self._get_epoch(state)
        if self._is_final_epoch(args, epoch):
            return True
        return epoch is not None and epoch % self.n == 0

    @staticmethod
    def _enable_eval_and_save(control):
        control.should_evaluate = True
        control.should_save = True
        return control

    @staticmethod
    def _disable_eval_and_save(control):
        control.should_evaluate = False
        control.should_save = False
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._should_run_on_epoch(args, state):
            return self._enable_eval_and_save(control)
        return self._disable_eval_and_save(control)


class WandbCustomLoggingCallback(TrainerCallback):
    """
    A custom callback to log specific metrics to WandB.
    NOTE: Only needed if you want logging beyond the default Trainer integration.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        # Check if WandB run has been initialized
        # (Trainer usually handles init if 'wandb' is in report_to)
        if state.is_world_process_zero and wandb.run is not None:
            if logs is not None and "loss" in logs:
                loss = logs["loss"]
                step = state.global_step

                # Log custom metric(s) to WandB
                wandb.log(
                    {
                        "custom_details/train_loss_per_log_step": loss,
                        # Add any other custom metrics you want here
                        # "custom_details/some_other_value": calculate_something()
                    },
                    step=step,
                )  # IMPORTANT: Use state.global_step for the x-axis
