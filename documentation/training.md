# Training

Run commands from the repository root after [installation](../README.md#installation)
and [data preparation](data_preparation.md).

Set `VLM_CONFIG_DIR` and `VLM_OUTPUT_DIR` as shown in the
[quick start](../README.md#quick-start), which also covers batch-size settings
and multi-GPU training. `--train_config` and `--add_base_configs` paths are relative
to `VLM_CONFIG_DIR`.

## nnFoundation

Start with the nnFoundationViT CT-RATE command in the
[quick start](../README.md#quick-start).

For nnFoundationCNN, use `nnFoundationCNN.yml` and its matching dataset base.
For Merlin, select the base under `bases/datasets/merlin/`.
These recipes train on `train_val`.

## RMR

First [prepare the RMR embeddings](rmr_embeddings.md).

Train Qwen3-VL-2B with Curia features and AveragePool at compression ratio 4:

```bash
export VLM_CONFIG_DIR="$PWD/paper_configs/RMR"

uv run python train.py \
  --train_config Compression/AveragePool/Qwen3-VL-2B__Curia-8S__AveragePool__CR_4.yml \
  --add_base_configs bases/datasets/ctrate/Curia-8S_ROI.yml \
  --trainer.per_device_train_batch_size 3 \
  --trainer.gradient_accumulation_steps 8
```

For Merlin, use `bases/datasets/merlin/Curia-8S_ROI.yml`. Each recipe uses the
`<encoder>_ROI` dataset base of either dataset, with these exceptions:

- `FOV` recipes for `Curia-8S` and `Primus-1B` also use the `_Whole` bases.
- `Primus-4B_2_1_2` is CT-RATE only and `Primus-4B_1_2_2` is Merlin only.
- `LLM/AveragePool` recipes are CT-RATE only.

[`runs.csv`](../paper_configs/RMR/runs.csv) lists the `--train_config` and
`--add_base_configs` pair of every paper run. Automatic post-training evaluation
uses `val`; use `--eval.split test` for held-out test evaluation.

## MIRAGE

MIRAGE recipes use CT-RATE v2 and a retrieval mapping per encoder. Generate the
mappings and train as described in [MIRAGE recipes](retrieval.md#mirage-recipes).

## Configuration overrides

Override config values with dotted flags, for example
`--trainer.per_device_train_batch_size 3`. CLI values override the run config;
the run config overrides its bases, with earlier bases taking precedence.
Use `${NAME}` in YAML paths to reference environment variables.
See [vlm/configs](../vlm/configs/) for all fields and defaults.

## Training phases

The paper recipes train in two phases. During `warmup_1_duration`, only the
adapter trains; afterwards, all configured parameters train with a fresh
optimizer, a second warmup over `warmup_2_duration`, and a cosine decay. Durations
are fractions of the whole training. The first phase is rounded to whole epochs
(for example, 0.1 of 12 epochs is 1 epoch) and must leave at least one epoch for
the second. A checkpoint is always saved at the end of the first phase. Phased
training requires `trainer.num_train_epochs` instead of `trainer.max_steps`.

## DeepSpeed

Pass a DeepSpeed config to the usual `torchrun` command, for example
`--trainer.deepspeed "$PWD/paper_configs/deepspeed/zero2.json"`. ZeRO-2
([zero2.json](../paper_configs/deepspeed/zero2.json)) shards gradients and
optimizer states across GPUs; ZeRO-3 ([zero3.json](../paper_configs/deepspeed/zero3.json))
also shards the model weights. Batch size, gradient accumulation, clipping, and
bf16 follow the trainer settings.

With bf16, DeepSpeed casts the whole model to bf16, so checkpoints store the
trained weights in bf16; inference casts them back. Checkpoints also contain
DeepSpeed's optimizer state for resuming, but not the frozen weights, which come
from the pretrained model.

## Resuming

Each training run creates a folder under `VLM_OUTPUT_DIR` containing its saved
configuration and checkpoints. For example:

```text
${VLM_OUTPUT_DIR}/ctrate/CTRATE_nnFoundationViT/
└── nnFoundationViT__2026-10-20_12-00-00__AB12CD34/
    ├── final_config.yml
    └── checkpoint-1000/
```

Recipes in subfolders add them to the path, for example
`${VLM_OUTPUT_DIR}/ctrate/CTRATE_Curia-8S_ROI/Compression/AveragePool/<run>/` for RMR.

Resume from the latest saved checkpoint using the run folder's absolute path:

```bash
uv run python train.py \
  --continue_training_run_dir "${VLM_OUTPUT_DIR}/ctrate/CTRATE_nnFoundationViT/nnFoundationViT__2026-10-20_12-00-00__AB12CD34"
```

Or pass just its folder name to search recursively under `VLM_OUTPUT_DIR`:

```bash
uv run python train.py \
  --continue_training_run_dir nnFoundationViT__2026-10-20_12-00-00__AB12CD34
```

The folder name must identify a unique run. Resuming restores its saved config
and training state; no `--train_config` is needed.

## Evaluation during training

nnFoundation and MIRAGE disable validation during training and evaluate the final
model on test. RMR evaluates CT-RATE every epoch and Merlin every two epochs, selecting
the best checkpoint by `eval_classifier_f1`.

Clinical finding scores use [RadBERT-CT](https://huggingface.co/IAMJB/RadBERT-CT)
for CT-RATE and [ModernBERT-Merlin](https://huggingface.co/MIC-DKFZ/ModernBERT-Merlin)
for Merlin.

For `trainer.eval_strategy: epoch`, set `eval.eval_every_n_epochs` to a positive
integer; the final epoch is always evaluated. For `steps`, use `trainer.eval_steps`.

Report generation uses a separate dataloader owned by `ReportEvaluationCallback`:

| Setting | Purpose |
| --- | --- |
| `eval.per_device_batch_size` | Reports generated per GPU per batch, during callback evaluation and inference; adjust to fit GPU memory |
| `eval.callback_n_samples` | Validation cases evaluated by the callback; `null` uses all, and `unique` selects one image version per CT-RATE scan |
| `trainer.per_device_eval_batch_size` | Batch size for the Trainer's validation-loss pass; defaults to `1` |
| `data.n_val_samples` | Cases used for the Trainer's validation-loss pass; defaults to `1` |

RMR selects checkpoints using the callback's classifier F1.
The two Trainer loss-pass settings do not limit the callback's evaluation dataset
or generation batch size. Configure `eval.per_device_batch_size` independently
of the effective training batch size of 24; generation uses no gradient accumulation.

Weights & Biases logging is disabled by default. Set `export WANDB_MODE=online`
to enable it, or `export WANDB_MODE=offline` to save logs locally for later syncing.
For offline model loading, pre-cache the language models and evaluation
classifiers, then set `export HF_HUB_OFFLINE=1` before launching Python.

See [inference and evaluation](inference_and_evaluation.md) for test reporting.
