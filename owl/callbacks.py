import re

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState

import wandb
from utils.filter_batch import extract_batch_prompt_only


class PrintEvalBatchCallback(TrainerCallback):
    def __init__(self, tokenizer=None, processor=None):
        self.tokenizer = tokenizer
        self.printed = False
        self.sample_count = 50

        self.last_epoch = -1

        self.max_diff = 0.2

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
        if state.epoch - self.last_epoch < self.max_diff:
            return
        self.last_epoch = state.epoch
        print(f"Output samples from validation dataset after epoch {state.epoch}:")
        print(f"Epoch {state.epoch} reached")

        n_correct_qa = 0

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
                del inputs_prompt_only["batch_count"]
                inputs_prompt_only["image_embeds"] = model.vision2text_model(
                    inputs_prompt_only["image_embeds"]
                ).to(torch.float16)
                outputs = model.language_model.generate(
                    **inputs_prompt_only,
                    max_new_tokens=256,  # Adjust max_new_tokens as needed
                )

                # Decode input and generated output
                decoded_output = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                decoded_gt = self.tokenizer.batch_decode(
                    inputs["input_ids"], skip_special_tokens=True
                )

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

        print(f"Correct QA: {n_correct_qa}/{self.sample_count}")

        wandb.log({"eval_table": self.eval_table})
        wandb.log({"correct_qa": n_correct_qa})

        model.train()  # Ensure model is back in training mode
