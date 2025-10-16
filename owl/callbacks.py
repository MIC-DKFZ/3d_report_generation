import os
import re
from typing import Literal
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import wandb
from transformers.modeling_utils import unwrap_model
from evaluation.metrics import bleu_score, rouge_score, meteor_score
from utils.filter_batch import extract_batch_prompt_only
from utils import logger
from utils.helper import get_trainable_state_dict
from utils.regex_patterns import VQA_ANSWER_PATTERN


class PrintEvalBatchCallback(TrainerCallback):
    def __init__(
        self, eval_dataloader: DataLoader, task_type: Literal["VQA", "RG"], max_new_tokens: int, sample_count: int,
            tokenizer=None, processor=None, eval_every_n_epochs=0.25, print_first_sample=True, log_n_per_callback=10
    ):
        self.eval_dataloader = eval_dataloader
        self.task_type = task_type
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self.sample_count = sample_count

        self.last_epoch = 0
        self.eval_every_n_epochs = eval_every_n_epochs

        self.print_first_sample = print_first_sample
        self.log_n_per_callback = min(self.sample_count, log_n_per_callback)

        # Only create wandb table if wandb is initialized (i.e., on main process)
        if wandb.run is not None:
            columns = [
                "Epoch",
                "Sample Index",
                "Generated Output",
                "Ground Truth"
            ]
            if self.task_type == "VQA":
                columns += [
                    "Answer Pred",
                    "Answer GT"
                ]
            self.eval_table = wandb.Table(columns=columns, log_mode="MUTABLE")

            logger.info(f"Created an eval table with the following columns: {columns}")
        self.processor = processor

        # Track best metrics for model saving
        self.best_rouge_score = -1.0    # RG
        self.best_accuracy = -1.0       # VQA

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

        if (not state.is_world_process_zero
                or state.epoch - self.last_epoch < self.eval_every_n_epochs):
            return

        self.last_epoch = state.epoch
        logger.info(f"Output samples from validation dataset after epoch {state.epoch}:")
        logger.info(f"Epoch {state.epoch} reached")

        preds = []
        gts = []

        model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            for i, inputs in enumerate(tqdm(self.eval_dataloader, desc="Inference")):
                inputs_prompt_only = extract_batch_prompt_only(inputs)
                inputs_prompt_only = inputs_prompt_only.to(args.device)
                del inputs_prompt_only["raw_samples"]
                del inputs_prompt_only["batch_count"]

                outputs = model.generate(
                    **inputs_prompt_only,
                    max_new_tokens=self.max_new_tokens,  # Adjust max_new_tokens as needed
                )

                # Decode input and generated output
                decoded_gts = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
                decoded_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Explicitly delete large tensors to free memory
                del inputs_prompt_only
                del outputs
                torch.cuda.empty_cache()  # Clear CUDA cache periodically

                if self.print_first_sample and i == 0:
                    logger.error(f"First batch sample GROUNDTRUTH:\n{decoded_gts[0]}")
                    logger.error(f"First batch sample PREDICTION:\n{decoded_preds[0]}")

                decoded_gt_assistant_answers = [decoded_gt.split("assistant\n")[-1].strip() for decoded_gt in decoded_gts]
                decoded_pred_assistant_answers = [decoded_output.split("assistant\n")[-1].strip() for decoded_output in decoded_preds]

                # store predictions and ground truths
                preds.extend(decoded_pred_assistant_answers)
                gts.extend(decoded_gt_assistant_answers)

        answer_preds = []
        answer_gts = []

        if self.task_type == "VQA":
            # === Count correct QA ===
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

            logger.info(f"Correct QA: {n_correct_qa}/{self.sample_count}")
            accuracy = n_correct_qa / self.sample_count
            wandb.log({
                "correct_qa": n_correct_qa,
                "accuracy": accuracy
            })

            # === Log and optionally save best model ===
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                logger.info(f"New best ACCURACY: {self.best_accuracy:.4f} - Saving model checkpoint")
                self.save_checkpoint(model, args.output_dir)

            # Always log best, even if it did not improve
            wandb.log({"best_accuracy": self.best_accuracy})

        elif self.task_type == "RG":
            # === BLEU ===
            max_orders = [1, 2, 3, 4]
            try:
                bleu_scores = {order: [] for order in max_orders}

                for pred, gt in zip(decoded_pred_assistant_answers, decoded_gt_assistant_answers):
                    for order in max_orders:
                        score = bleu_score(pred=[pred], gt=[gt], max_order=order)["bleu"]
                        bleu_scores[order].append(score)

                # Compute average per BLEU-n
                bleu_averages = {order: np.mean(scores).item() for order, scores in bleu_scores.items()}

            except Exception as e:
                logger.error(f"Bleu score calculation failed: {e}")
                bleu_averages = {order: -1 for order in max_orders}

            # Log the BLEU scores
            for order, score in bleu_averages.items():
                logger.info(f"bleu_score_{order}: {score:.4f}")
                wandb.log({f"bleu_score_{order}": score})

            # === METEOR ===
            try:
                meteor_score_val = meteor_score(preds, gts)["meteor"]
            except Exception as e:
                logger.error(f"METEOR score calculation failed: {e}")
                meteor_score_val = -1  # Use -1 to indicate failure in logging

            # Log the METEOR score
            logger.info(f"METEOR score: {meteor_score_val}")
            wandb.log({"meteor_score": meteor_score_val})

            # === ROUGE ===
            try:
                rouge_score_val = rouge_score(preds, gts)["rougeL"]
            except Exception as e:
                logger.error(f"ROUGE score calculation failed: {e}")
                rouge_score_val = -1

            # Log the ROUGE score
            logger.info(f"ROUGE score: {rouge_score_val}")
            wandb.log({"rouge_score": rouge_score_val})

            # === Log and optionally save best model ===
            if rouge_score_val > self.best_rouge_score:
                self.best_rouge_score = rouge_score_val
                logger.info(f"New best ROUGE score: {self.best_rouge_score:.4f} - Saving model checkpoint")
                self.save_checkpoint(model, args.output_dir)

            # Always log best, even if it did not improve
            wandb.log({"best_rouge_score": self.best_rouge_score})
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        # Only log 'self.log_n_per_callback' number of cases to the WandB eval table
        for idx in range(0, self.sample_count, self.sample_count // self.log_n_per_callback):
            table_data = [
                state.epoch,
                idx,
                preds[idx],
                gts[idx]
            ]
            if self.task_type == "VQA":
                table_data += [
                    answer_preds[idx],
                    answer_gts[idx]
                ]
            self.eval_table.add_data(*table_data)

        wandb.log({"eval_table": self.eval_table})

        model.train()  # Ensure model is back in training mode

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
        logger.info(f"Model saved to: {best_model_dir}")



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
