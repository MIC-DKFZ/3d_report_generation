"""Training loss and optimizer learning-rate logging."""

from torch.optim import Optimizer
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import wandb

from vlm.utils import logger


class LossLoggingCallback(TrainerCallback):
    """Write training and validation losses to the main process log."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if not state.is_world_process_zero or logs is None:
            return
        if "loss" in logs:
            logger.log("LOSS", f"Step: {state.global_step} | Training Loss: {logs['loss']:.5f}")
        elif "eval_loss" in logs:
            logger.log(
                "LOSS", f"Step: {state.global_step} | Validation Loss: {logs['eval_loss']:.5f}"
            )


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
