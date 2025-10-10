import os
import re
from tqdm import tqdm
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import wandb
from transformers.modeling_utils import unwrap_model

from evaluation.metrics import bleu_score, rouge_score, meteor_score
from owl.model import OwnOwlModel
from utils.filter_batch import extract_batch_prompt_only
from utils import logger
from utils.helper import get_trainable_state_dict


class PrintEvalBatchCallback(TrainerCallback):
    def __init__(
        self, tokenizer=None, processor=None, sample_count=250, eval_every_n_epochs=0.25, print_first_sample=True
    ):
        self.tokenizer = tokenizer
        self.printed = False
        self.sample_count = sample_count

        self.last_epoch = 0
        self.eval_every_n_epochs = eval_every_n_epochs

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

        self.meteor_failed = False

        # Track best rouge score for model saving
        self.best_rouge_score = -1.0

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

        # if metrics is not None:
        #     metrics["eval_rouge_score"] = -1  # Initialize to -1 in case of failure

        if (not state.is_world_process_zero
                or state.epoch - self.last_epoch < self.eval_every_n_epochs):
            return

        self.last_epoch = state.epoch
        logger.info(f"Output samples from validation dataset after epoch {state.epoch}:")
        logger.info(f"Epoch {state.epoch} reached")

        n_correct_qa = 0

        model_str = "owl" if isinstance(model, OwnOwlModel) else "qwen"

        # metrics_dict = defaultdict(list)
        preds = []
        gts = []

        # Only create wandb table if wandb is initialized (i.e., on main process)
        if wandb.run is not None:
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
        else:
            self.eval_table = None
        model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            for i, inputs in enumerate(tqdm(eval_dataloader, total=self.sample_count)):
                if i >= self.sample_count:  # Limit the number of samples to print
                    break
                inputs_prompt_only = extract_batch_prompt_only(inputs)
                inputs_prompt_only = inputs_prompt_only.to(args.device)
                del inputs_prompt_only["raw_samples"]
                del inputs_prompt_only["batch_count"]

                if model_str == "owl":
                    # Process embeddings with explicit memory management
                    embeddings = model.vision2text_model(inputs_prompt_only["image_embeds"])
                    inputs_prompt_only["image_embeds"] = embeddings.to(torch.float16)
                    del embeddings  # Explicitly delete intermediate tensor

                    outputs = model.language_model.generate(
                        **inputs_prompt_only,
                        max_new_tokens=640,  # Adjust max_new_tokens as needed
                    )
                else:
                    outputs = model.generate(
                        **inputs_prompt_only,
                        max_new_tokens=640,  # Adjust max_new_tokens as needed
                    )

                # Decode input and generated output
                decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_gt = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

                # Explicitly delete large tensors to free memory
                del inputs_prompt_only
                del outputs
                torch.cuda.empty_cache()  # Clear CUDA cache periodically

                if self.print_first_sample and i == 0:
                    logger.error(f"First sample input: {decoded_gt[0]}")
                    logger.error(f"First sample output: {decoded_output[0]}")

                decoded_gt_assistant_answer = (
                    decoded_gt[0].split("assistant\n")[-1].strip().split("assistant\n")[0].strip()
                )
                if model_str == "owl":
                    try:
                        decoded_gt_assistant_answer = re.search(
                            r"'text':\s*'([^']*)'", decoded_gt_assistant_answer
                        ).group(1)
                    except Exception:
                        pass

                decoded_pred_assistant_answer = ""
                try:
                    decoded_pred_assistant_answer = decoded_output[0].split("assistant\n")[-1].strip()
                    if model_str == "owl":
                        decoded_pred_assistant_answer = re.search(
                            r"'text':\s*'([^']*)'", decoded_pred_assistant_answer
                        ).group(1)
                except Exception:
                    pass

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

        try:

            # Log the Bleu score
            bleu_score_1 = bleu_score(preds, gts, max_order=1)["bleu"]
            bleu_score_2 = bleu_score(preds, gts, max_order=2)["bleu"]
            bleu_score_3 = bleu_score(preds, gts, max_order=3)["bleu"]
            bleu_score_4 = bleu_score(preds, gts, max_order=4)["bleu"]
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

        except Exception as e:
            logger.error(f"Bleu score calculation failed: {e}")
            bleu_score_1 = -1  # Use -1 to indicate failure in logging
            bleu_score_2 = -1
            bleu_score_3 = -1
            bleu_score_4 = -1

        if not self.meteor_failed:
            try:
                # Log the METEOR score
                meteor_score_val = meteor_score(preds, gts)["meteor"]
            except Exception as e:
                logger.error(f"METEOR score calculation failed: {e}")
                self.meteor_failed = True
                meteor_score_val = -1  # Use -1 to indicate failure in logging

            # # Log the METOR score
            # meteor_score_val = meteor_score(preds, gts)["meteor"]
            logger.info(f"METEOR score: {meteor_score_val}")
            wandb.log({"meteor_score": meteor_score_val})

        try:

            # Log the ROUGE score
            rouge_score_val = rouge_score(preds, gts)["rougeL"]
            logger.info(f"ROUGE score: {rouge_score_val}")
            wandb.log({"rouge_score": rouge_score_val})

        except Exception as e:
            logger.error(f"ROUGE score calculation failed: {e}")
            rouge_score_val = -1

        if metrics is not None:
            metrics["eval_rouge_score"] = float(rouge_score_val)

        logger.info(f"Correct QA: {n_correct_qa}/{self.sample_count}")

        wandb.log({"eval_table": self.eval_table})
        wandb.log({"correct_qa": n_correct_qa})

        # Save model if we found a new best rouge score
        if rouge_score_val > self.best_rouge_score:
            self.best_rouge_score = rouge_score_val
            logger.info(f"New best ROUGE score: {rouge_score_val:.4f} - Saving model checkpoint")

            # Create best model directory
            best_model_dir = os.path.join(args.output_dir, "best_model")
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

            # Log to wandb
            wandb.log({"best_rouge_score": self.best_rouge_score})
            logger.info(f"Model saved to: {best_model_dir}")

        model.train()  # Ensure model is back in training mode

        # Clean up memory after evaluation
        # del preds
        # del gts
        # torch.cuda.empty_cache()
        # import gc

        # gc.collect()


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
