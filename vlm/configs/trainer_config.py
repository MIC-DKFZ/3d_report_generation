from typing import Literal
from pydantic import Field
from vlm.configs.base import ConfigBase


class SFTTrainerConfig(ConfigBase):
    """
    Configuration for all arguments that are to be passed on to the SFTTrainer class including our default values.
    """
    num_train_epochs: int = Field(3, ge=1, description="Number of training epochs to run.")
    per_device_train_batch_size: int = Field(1, ge=1, description="Batch size per device during training.")
    per_device_eval_batch_size: int = Field(1, ge=1, description="Batch size per device during evaluation.")

    gradient_accumulation_steps: int = Field(1, ge=1, description="Number of updates steps to accumulate before performing a backward/update pass.")
    gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing to save memory at the cost of extra compute.")
    gradient_checkpointing_kwargs: dict = Field(default_factory=lambda: {"use_reentrant": False}, description="Additional keyword arguments for gradient checkpointing.")

    optim: str = Field("adamw_torch", description="Optimizer to use (e.g. adamw_torch, adamw_torch_fused, adamw_hf).")
    learning_rate: float = Field(2e-4, gt=0, description="Initial learning rate for optimizer.")
    lr_scheduler_type: str = Field("cosine", description="Learning rate scheduler type (e.g. linear, cosine, polynomial).")
    warmup_ratio: float = Field(0.03, ge=0, le=1, description="Ratio of total steps used for linear learning rate warmup.")
    max_grad_norm: float = Field(0.3, gt=0, description="Maximum gradient norm for gradient clipping.")

    logging_steps: int = Field(25, ge=1, description="Number of steps between logging events.")

    eval_steps: int = Field(250, ge=1, description="Number of steps between evaluation runs.")
    eval_strategy: Literal["no", "steps", "epoch"] = Field("steps", description="Evaluation strategy to adopt during training.")
    metric_for_best_model: str = Field("eval_loss", description="Metric to use for selecting the best model (e.g. eval_loss, accuracy).")
    greater_is_better: bool = Field(False, description="Whether the chosen metric should be maximized (True) or minimized (False).")

    save_steps: int = Field(250, ge=1, description="Save checkpoint every X steps when using save_strategy='steps'.")
    save_strategy: Literal["no", "steps", "epoch"] = Field("steps", description="Checkpoint saving strategy during training.")
    save_total_limit: int = Field(2, description="Maximum number of checkpoints to keep. Older ones are deleted.")

    load_best_model_at_end: bool = Field(False, description="Enables saving of best checkpoint if True! (And whether to load the best model found during training at the end). "
                                                            "WARNING: Enabling this may cause an NCCL desync error at the end of training.")
    dataset_kwargs: dict = Field(default_factory=lambda: {"skip_prepare_dataset": True}, description="Extra keyword arguments passed to dataset loading/preparation.")
    remove_unused_columns: bool = Field(False, description="Whether to remove unused columns from the dataset before training.")
    report_to: list[str] = Field(default_factory=lambda: ["wandb"], description="List of integrations to report training logs to (e.g. wandb, tensorboard).")

    dataloader_num_workers: int = Field(2, description="Number of subprocesses to use for loading data in parallel.")
    dataloader_pin_memory: bool = Field(True, description="Whether to copy tensors into CUDA pinned memory for faster GPU transfers.")
    dataloader_prefetch_factor: int = Field(1, description="Reduce prefetch factor to minimize memory usage.")
    dataloader_persistent_workers: bool = Field(False, description="Disable persistent workers to prevent memory accumulation.")
    torch_empty_cache_steps: int = Field(10, description="Clear PyTorch CUDA cache every X steps to manage GPU memory.")
    ddp_find_unused_parameters: bool = Field(False, description="Whether to enable detection of unused parameters in DDP training.")
    ddp_bucket_cap_mb: int = Field(25, description="Bucket size in MB for gradient synchronization in DDP.")
