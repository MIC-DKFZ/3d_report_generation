import math
import json
from pathlib import Path
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import wandb
from vlm.utils.decoding import decode_generated_answers, decode_label_answers
from vlm.utils.ddp import gather_inference_outputs
from vlm.data.batches import extract_batch_prompt_only
from vlm.utils import logger
from vlm.utils.ddp import is_main_process
from vlm.utils.serialization import save_json

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


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

        self.radbert_scorer = None
        if calculate_radbert_score:
            from vlm.registries.evaluation_registry import get_classifier_evaluator

            self.radbert_scorer = get_classifier_evaluator(dataset_name)
        self.calculate_radbert_score = self.radbert_scorer is not None

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
    """
    Run evaluation every ``n`` epochs, using ``trainer.eval_steps`` as ``n``.
    This enables an epoch-based evaluation cadence other than every epoch.
    """

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
