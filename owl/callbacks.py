from collections import defaultdict
import os
from pathlib import Path
import re

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import wandb
from Eval.metrics import BleuScoreHF, METEORScore, ROUGEScore
from utils.filter_batch import extract_batch_prompt_only
from utils import logger

from models.owl_vlm import OwlVLModel
from models.qwen_vlm import QwenVLModel


class PrintEvalBatchCallback(TrainerCallback):
    def __init__(
        self, tokenizer=None, processor=None, sample_count=50, eval_every_n_epochs=0.2, print_first_sample=True
    ):
        self.tokenizer = tokenizer
        self.printed = False
        self.sample_count = sample_count

        self.last_epoch = -1

        self.max_diff = eval_every_n_epochs

        self.print_first_sample = print_first_sample

        self.eval_table = wandb.Table(
            columns=[
                "Epoch",
                "Sample Index",
                "Generated Output",
                "Ground Truth",
                "Answer Pred",
                "Answer GT",
            ]
        )

        self.processor = processor

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
        # TODO: for DDP check something like trainer.is_world_process_zero() here
        if state.epoch - self.last_epoch < self.max_diff:
            return
        self.last_epoch = state.epoch
        logger.info(f"Output samples from validation dataset after epoch {state.epoch}:")
        logger.info(f"Epoch {state.epoch} reached")

        n_correct_qa = 0

        model_str = "owl" if isinstance(model, OwlVLModel) else "qwen"

        # metrics_dict = defaultdict(list)
        preds = []
        gts = []

        self.eval_table = wandb.Table(
            columns=[
                "Epoch",
                "Sample Index",
                "Generated Output",
                "Ground Truth",
                "Answer Pred",
                "Answer GT",
            ]
        )
        model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            for i, inputs in enumerate(eval_dataloader):
                if i >= self.sample_count:  # Limit the number of samples to print
                    break
                inputs_prompt_only = extract_batch_prompt_only(inputs)
                inputs_prompt_only = inputs_prompt_only.to(args.device)
                del inputs_prompt_only["raw_samples"]
                del inputs_prompt_only["batch_count"]

                if model_str == "owl":
                    inputs_prompt_only["image_embeds"] = model.vision2text_model(
                        inputs_prompt_only["image_embeds"]
                    ).to(torch.float16)
                    outputs = model.language_model.generate(
                        **inputs_prompt_only,
                        max_new_tokens=256,  # Adjust max_new_tokens as needed
                    )
                else:
                    outputs = model.generate(
                        **inputs_prompt_only,
                        max_new_tokens=256,  # Adjust max_new_tokens as needed
                    )

                # Decode input and generated output
                decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_gt = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

                if self.print_first_sample and i == 0:
                    logger.info(f"First sample input: {decoded_gt[0]}")
                    logger.info(f"First sample output: {decoded_output[0]}")

                decoded_gt_assistant_answer = (
                    decoded_gt[0].split("assistant\n")[-1].strip().split("assistant\n")[0].strip()
                )
                if model_str == "owl":
                    decoded_gt_assistant_answer = re.search(r"'text':\s*'([^']*)'", decoded_gt_assistant_answer).group(
                        1
                    )

                try:
                    decoded_pred_assistant_answer = decoded_output[0].split("assistant\n")[-1].strip()
                    if model_str == "owl":
                        decoded_pred_assistant_answer = re.search(
                            r"'text':\s*'([^']*)'", decoded_pred_assistant_answer
                        ).group(1)
                except Exception:
                    decoded_pred_assistant_answer = ""

                # store predictions and ground truths
                preds.append(decoded_pred_assistant_answer)
                gts.append(decoded_gt_assistant_answer)

                m = re.search(r"The answer is (.)", decoded_output[0])
                if m:
                    answer_pred = m.group(0)
                else:
                    answer_pred = "No answer found"

                m = re.search(r"The answer is (.)", decoded_gt[0])
                if m:
                    answer_gt = m.group(0)
                else:
                    answer_gt = "No answer found"

                if answer_pred == answer_gt:
                    n_correct_qa += 1

                self.eval_table.add_data(
                    state.epoch,
                    i,
                    str(decoded_output[0]),
                    str(decoded_gt[0]),
                    answer_pred,
                    answer_gt,
                )

        # Log the Bleu score
        bleu_score_1 = BleuScoreHF(preds, gts, max_order=1)["bleu"]
        bleu_score_2 = BleuScoreHF(preds, gts, max_order=2)["bleu"]
        bleu_score_3 = BleuScoreHF(preds, gts, max_order=3)["bleu"]
        bleu_score_4 = BleuScoreHF(preds, gts, max_order=4)["bleu"]
        logger.info(f"Bleu1 score: {bleu_score_1}")
        logger.info(f"Bleu2 score: {bleu_score_2}")
        logger.info(f"Bleu3 score: {bleu_score_3}")
        logger.info(f"Bleu4 score: {bleu_score_4}")

        # Log the Bleu scores to WandB
        wandb.log(
            {
                "bleu_score_1": bleu_score_1,
                "bleu_score_2": bleu_score_2,
                "bleu_score_3": bleu_score_3,
                "bleu_score_4": bleu_score_4,
            }
        )

        # Log the METOR score
        meteor_score = METEORScore(preds, gts)["meteor"]
        logger.info(f"METEOR score: {meteor_score}")
        wandb.log({"meteor_score": meteor_score})

        # Log the ROUGE score
        rouge_score = ROUGEScore(preds, gts)["rougeL"]
        logger.info(f"ROUGE score: {rouge_score}")
        wandb.log({"rouge_score": rouge_score})

        logger.info(f"Correct QA: {n_correct_qa}/{self.sample_count}")

        wandb.log({"eval_table": self.eval_table})
        wandb.log({"correct_qa": n_correct_qa})

        model.train()  # Ensure model is back in training mode


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
            # elif 'eval_loss' in logs:
            #     eval_loss = logs['eval_loss']
            #     global_step = state.global_step
            #     print(f"Step: {global_step} | Validation Loss: {eval_loss:.4f}")

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
