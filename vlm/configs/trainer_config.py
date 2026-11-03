from typing import Literal
from pydantic import Field, model_validator
from vlm.configs.base import ConfigBase
from vlm.configs.paths import ConfigPath


class TrainerConfig(ConfigBase):
    """
    Configuration for all arguments that are to be passed on to 'TrainingArguments' including our default values.
    """
    num_train_epochs: int = Field(3, ge=1, description="Number of training epochs to run.")
    max_steps: int = Field(-1, ge=-1, description="Number of steps to take during training.")
    per_device_train_batch_size: int = Field(1, ge=1, description="Batch size per device during training.")
    per_device_eval_batch_size: int = Field(1, ge=1, description="Batch size per device during evaluation.")

    gradient_accumulation_steps: int = Field(1, ge=1, description="Number of updates steps to accumulate before performing a backward/update pass.")
    average_tokens_across_devices: bool = Field(
        True,
        description="Normalize training loss by supervised tokens across all DDP processes.",
    )
    gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing to save memory at the cost of extra compute.")
    gradient_checkpointing_kwargs: dict = Field(default_factory=lambda: {"use_reentrant": False}, description="Additional keyword arguments for gradient checkpointing.")
    gradient_checkpointing_fraction: float | None = Field(
        None,
        gt=0,
        le=1,
        description=(
            "Fraction of the decoder layers to checkpoint, spread evenly over the model; the others "
            "keep their activations. 'None' leaves the choice to 'gradient_checkpointing_kwargs'."
        ),
    )

    max_grad_norm: float = Field(0.3, gt=0, description="Maximum gradient norm for gradient clipping.")

    logging_steps: int = Field(25, ge=1, description="Number of steps between logging events.")

    eval_steps: int | None = Field(1, ge=1, description="Number of steps between evaluations when eval_strategy='steps'.")
    eval_strategy: Literal["no", "steps", "epoch"] | None = Field("epoch", description="Evaluation strategy to adopt during training.")
    metric_for_best_model: str | None = Field("eval_loss", description="Metric to use for selecting the best model (e.g. 'eval_loss', 'accuracy').")
    greater_is_better: bool = Field(False, description="Whether the chosen metric should be maximized ('True') or minimized ('False').")
    early_stopping_patience: int | None = Field(None, ge=1, description="Number of evaluation calls without metric improvement before stopping. 'None' disables early stopping.")
    early_stopping_threshold: float = Field(0.0, ge=0.0, description="Minimum metric improvement required to reset early stopping patience.")

    save_steps: int = Field(1, ge=1, description="Save checkpoint every X steps when 'save_strategy' is 'steps'.")
    save_strategy: Literal["no", "steps", "epoch"] = Field("epoch", description="Checkpoint saving strategy during training.")
    save_total_limit: int = Field(2, description="Maximum number of checkpoints to keep. Older ones are deleted.")

    load_best_model_at_end: bool = Field(False, description="Enables saving of best checkpoint if 'True'! (And whether to load the best model found during training at the end). "
                                                            "WARNING: Enabling this may cause an NCCL desync error at the end of training.")
    remove_unused_columns: bool = Field(False, description="Whether to remove unused columns from the dataset before training.")
    report_to: list[str] = Field(default_factory=lambda: ["wandb"], description="List of integrations to report training logs to (e.g. 'wandb', 'tensorboard').")

    dataloader_num_workers: int | None = Field(
        None,
        description=(
            "Number of subprocesses per device for data loading during training, validation, "
            "and inference. 'None' calculates the count during training setup."
        ),
    )
    dataloader_pin_memory: bool = Field(True, description="Whether to copy tensors into CUDA pinned memory for faster GPU transfers.")
    dataloader_prefetch_factor: int = Field(2, description="Reduce prefetch factor to minimize memory usage.")
    dataloader_persistent_workers: bool = Field(False, description="Disable persistent workers to prevent memory accumulation.")
    torch_empty_cache_steps: int = Field(10, description="Clear PyTorch CUDA cache every X steps to manage GPU memory.")
    ddp_find_unused_parameters: bool = Field(False, description="Whether to enable detection of unused parameters in DDP training.")
    ddp_bucket_cap_mb: int = Field(25, description="Bucket size in MB for gradient synchronization in DDP.")
    deepspeed: ConfigPath | None = Field(
        None,
        description=(
            "DeepSpeed config JSON (e.g. '/path/to/zero2.json'); 'None' trains with "
            "DDP."
        ),
    )

    @model_validator(mode="after")
    def validate_gradient_checkpointing_fraction(self):
        if self.gradient_checkpointing_fraction is None:
            return self
        if not self.gradient_checkpointing:
            raise ValueError("'gradient_checkpointing_fraction' requires 'gradient_checkpointing'.")
        if "every_n_layers" in self.gradient_checkpointing_kwargs:
            raise ValueError(
                "Set either 'gradient_checkpointing_fraction' or "
                "'gradient_checkpointing_kwargs.every_n_layers', not both."
            )
        return self
