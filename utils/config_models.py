from datetime import datetime
from typing import Optional, Literal
from uuid import uuid4
from pydantic import BaseModel, Field


class CoreConfig(BaseModel):
    """
    Configuration for all arguments that must be provided via CLI.
    """
    train_config: str = Field(..., description="Path to config file (YAML or JSON).")
    dataset_name: Literal["amosmm", "ctrate"] = Field(..., description="Name of the dataset.")
    dataset_path: str = Field(..., description="Path to the dataset JSON file directory.")
    embeddings_path: str = Field(..., description="Path to the vision embeddings.")


class GeneralConfig(BaseModel):
    """
    Configuration for general training setup, including but not limited to:
        - model configuration
        - data paths and flags
        - number of train/eval samples
    """
    model_type: Literal["owl", "qwen"] = Field(..., description="Type of VLM model to train.")
    model_id: str = Field(..., description="Model ID or path.")

    run_name: str = Field(datetime.now().strftime(f"run_%Y-%m-%d_%H-%M-%S_{uuid4()}"),
                          description="Name of the run. Also applies naming to wandb.")
    vqa_path: str = Field(..., description="Path to generated VQA data.")
    output_dir: str = Field("output/vlm", description="Directory to save model outputs.")
    wandb_project: str = Field("mic-rocket-vlm", description="WandB project name.")

    n_train_samples: int | None = Field(None, description="Number of training samples to use.")
    n_val_samples: int = Field(50, description="Number of validation samples to use (default: 50).")
    embedding_constant: float | Literal["random"] | None = Field(
        None,
        description="Constant value for language-only baseline. Either a float or 'random'."
    )
    train_only_vision_adapter: bool = Field(
        False,
        description="Train only the vision adapter (Qwen only)."
    )
    train_vision_encoder: bool = Field(
        False,
        description="Train the vision encoder instead of using frozen embeddings."
    )
    freeze_vision_encoder: bool = Field(
        False,
        description="Freeze the vision encoder (no LoRA adapter)."
    )
    vision_encoder_path: str | None = Field(None, description="Path to vision encoder checkpoint.")

    split_file: str | None = Field(None, description="Path to cross validation split file (JSON).")
    split_idx: int | None = Field(None, description="Index of the split to use (0-4).")
    dataset_cache: str | None = Field(None, description="Path to cache file for the dataset.")
    dataset_cache_val: str | None = Field(None, description="Path to validation cache file for the dataset.")
    use_csv_reports: bool = Field(False, description="Get reports from CSV file instead of YAML files.")
    do_resize: bool = Field(False, description="Resize images to 192x192x192 cubes (with lung crops).")
    do_transforms: bool = Field(False, description="Enable nnUNetv2 image augmentation.")
    local_files_only: bool = Field(False, description="Use only local files for Hugging Face from_pretrained.")

    # TODO: move this to a specific VLM adapter BaseModel
    reduce_block: list[int] = Field(
        default_factory=lambda: [2, 2, 2],
        description="List for reduce_block, e.g. [2,2,2] or [4,4,4]."
    )


class SFTTrainerConfig(BaseModel):
    """
    Configuration for all arguments that are to be passed on to the SFTTrainer class.
    """
    num_train_epochs: int = Field(3, ge=1, description="Number of training epochs to run.")
    per_device_train_batch_size: int = Field(1, ge=1, description="Batch size per device during training.")
    per_device_eval_batch_size: int = Field(1, ge=1, description="Batch size per device during evaluation.")
    gradient_accumulation_steps: int = Field(1, ge=1, description="Number of updates steps to accumulate before performing a backward/update pass.")
    gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing to save memory at the cost of extra compute.")

    optim: str = Field("adamw_torch", description="Optimizer to use (e.g. adamw_torch, adamw_torch_fused, adamw_hf).")
    learning_rate: float = Field(2e-4, gt=0, description="Initial learning rate for optimizer.")
    lr_scheduler_type: str = Field("cosine", description="Learning rate scheduler type (e.g. linear, cosine, polynomial).")

    logging_steps: int = Field(10, ge=1, description="Number of steps between logging events.")
    eval_steps: int = Field(250, ge=1, description="Number of steps between evaluation runs.")
    eval_strategy: Literal["no", "steps", "epoch"] = Field("steps", description="Evaluation strategy to adopt during training.")
    metric_for_best_model: str = Field("eval_loss", description="Metric to use for selecting the best model (e.g. eval_loss, accuracy).")
    greater_is_better: bool = Field(False, description="Whether the chosen metric should be maximized (True) or minimized (False).")

    save_strategy: Literal["no", "steps", "epoch"] = Field("steps", description="Checkpoint saving strategy during training.")
    save_steps: int = Field(500, ge=1, description="Save checkpoint every X steps when using save_strategy='steps'.")
    save_total_limit: int = Field(2, description="Maximum number of checkpoints to keep. Older ones are deleted.")

    load_best_model_at_end: bool = Field(True, description="Whether to load the best model found during training at the end.")
    max_grad_norm: float = Field(0.3, gt=0, description="Maximum gradient norm for gradient clipping.")
    warmup_ratio: float = Field(0.03, ge=0, le=1, description="Ratio of total steps used for linear learning rate warmup.")
    gradient_checkpointing_kwargs: dict = Field(default_factory=lambda: {"use_reentrant": False}, description="Additional keyword arguments for gradient checkpointing.")
    dataset_kwargs: dict = Field(default_factory=lambda: {"skip_prepare_dataset": True}, description="Extra keyword arguments passed to dataset loading/preparation.")
    remove_unused_columns: bool = Field(False, description="Whether to remove unused columns from the dataset before training.")
    report_to: list[str] = Field(default_factory=lambda: ["wandb"], description="List of integrations to report training logs to (e.g. wandb, tensorboard).")

    torch_empty_cache_steps: int = Field(10, description="Clear PyTorch CUDA cache every X steps to manage GPU memory.")
    ddp_find_unused_parameters: bool = Field(False, description="Whether to enable detection of unused parameters in DDP training.")
    ddp_bucket_cap_mb: int = Field(25, description="Bucket size in MB for gradient synchronization in DDP.")


class VLMConfig(BaseModel):
    """
    Configuration for a Vision-Language Model
    """
    model_id: str
    model_class: type