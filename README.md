# VLM: 3D Radiology Report Generation

[![nnFoundation paper](https://img.shields.io/badge/nnFoundation-arXiv-b31b1b?logo=arxiv)](https://arxiv.org/abs/2609.26924)
[![nnFoundationViT models](https://img.shields.io/badge/Hugging%20Face-nnFoundationViT-ffc107?logo=huggingface)](https://huggingface.co/MIC-DKFZ/nnFoundationViT)
[![nnFoundationCNN models](https://img.shields.io/badge/Hugging%20Face-nnFoundationCNN-ffc107?logo=huggingface)](https://huggingface.co/MIC-DKFZ/nnFoundationCNN)

[![RMR paper](https://img.shields.io/badge/RMR-arXiv-b31b1b?logo=arxiv)](https://arxiv.org/abs/2608.08713)
[![ModernBERT-Merlin model](https://img.shields.io/badge/Hugging%20Face-ModernBERT--Merlin-ffc107?logo=huggingface)](https://huggingface.co/MIC-DKFZ/ModernBERT-Merlin)

A framework for radiology report generation from 3D medical images, supporting
training, inference, and evaluation of vision-language models.

| Project | Paper | Training configurations |
| --- | --- | --- |
| **nnFoundation** | [nnFoundation: 3D Foundation Models for Radiology](https://arxiv.org/abs/2609.26924) | [paper_configs/nnFoundation](paper_configs/nnFoundation/) |
| **RMR** | [Resolution Meets Reduction: Efficient Visual Context for 3D Radiology Report Generation](https://arxiv.org/abs/2608.08713) | [paper_configs/RMR](paper_configs/RMR/) |
| **MIRAGE** | Coming soon | [paper_configs/MIRAGE](paper_configs/MIRAGE/) |

**nnFoundation:** This repository provides the downstream report-generation
experiments for nnFoundation. Frozen features from
[nnFoundationViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT) or
[nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN) are connected
to Qwen2.5-VL-3B through a trainable vision-to-language adapter, with LoRA
adaptation of the language model.

**RMR (Resolution Meets Reduction):** RMR investigates how to preserve useful
visual detail while reducing the token cost of 3D report generation. Its
experiments compare input field of view and resolution, visual token compression
methods, and language-model choice on CT-RATE and Merlin.

**MIRAGE:** MIRAGE adds retrieval to report generation on CT-RATE. Reports, or
an image and its report, from the nearest training cases in vision-embedding
space are added to the prompt of Qwen3-VL-2B. See
[retrieval mappings](documentation/retrieval.md#mirage-recipes) for the recipes.

## Installation

Use Python **3.12**, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a CUDA-capable GPU. Standalone evaluation also requires Java for METEOR.
From the repository root:

```bash
uv sync --locked --python 3.12
```

## Quick start

The example below trains nnFoundationViT on CT-RATE. Before running it,
[prepare the dataset JSONs and vision embeddings](documentation/data_preparation.md)
and export the config's data-path variables, `CTRATE_DATASET_JSONS` and `VLM_EMBEDS`,
as described there.

The framework reads two environment variables directly:

| Variable | Purpose |
| --- | --- |
| `VLM_CONFIG_DIR` | Root for resolving configuration files |
| `VLM_OUTPUT_DIR` | Root for saving runs and finding existing runs by folder name |

The experiments use an **effective batch size of 24**. Set the per-GPU batch size
and gradient accumulation for your available GPU memory and GPU count:

```text
effective batch size = number of GPUs × per-GPU batch size × gradient accumulation steps
```

For example, train nnFoundationViT on CT-RATE with one GPU (`1 × 3 × 8 = 24`):

```bash
export VLM_CONFIG_DIR="$PWD/paper_configs/nnFoundation"
export VLM_OUTPUT_DIR="$PWD/output"

uv run python train.py \
  --train_config nnFoundationViT.yml \
  --add_base_configs bases/datasets/ctrate/nnFoundationViT.yml \
  --trainer.per_device_train_batch_size 3 \
  --trainer.gradient_accumulation_steps 8
```

The example uses `$PWD/output`. You can set `VLM_OUTPUT_DIR` to any writable
location, including outside the repository, with enough space for checkpoints.

Multi-GPU training uses **Distributed Data Parallel (DDP)** through `torchrun`.
For four GPUs (`4 × 3 × 2 = 24`):

```bash
uv run torchrun --standalone --nproc_per_node=4 train.py \
  --train_config nnFoundationViT.yml \
  --add_base_configs bases/datasets/ctrate/nnFoundationViT.yml \
  --trainer.per_device_train_batch_size 3 \
  --trainer.gradient_accumulation_steps 2
```

Set `--eval.per_device_batch_size` separately to fit report generation in GPU memory.
See [evaluation settings](documentation/training.md#evaluation-during-training) for
the callback and Trainer batch-size distinction.

This nnFoundation recipe saves its configuration and checkpoints under
`VLM_OUTPUT_DIR`, then automatically generates reports and evaluates the final
checkpoint on the test set. See [inference and evaluation](documentation/inference_and_evaluation.md)
for the output files and directory structure.

## Reproducing nnFoundation and RMR experiments

The nnFoundation and RMR experiments initialized the adapter's linear layers with
`self.model.post_init()` (zero-mean normal weights with Qwen's `initializer_range`,
zero biases). The default PyTorch initialization performs better, so the call is
disabled by default.

**To reproduce the paper results, uncomment `# self.model.post_init()`** in
the [Qwen25VL](vlm/models/qwen25vl/wrapper.py) or
[Qwen3VL](vlm/models/qwen3vl/wrapper.py) wrapper used by your configuration. Configurations with other language models are
not affected.

## Documentation

- [Data preparation and paths](documentation/data_preparation.md)
- [RMR embedding names and requirements](documentation/rmr_embeddings.md)
- [Training recipes, overrides, and resuming](documentation/training.md)
- [Inference and evaluation](documentation/inference_and_evaluation.md)
- [Retrieval mappings](documentation/retrieval.md)

## Citation

Please cite the paper corresponding to the configurations or methods you use.

```bibtex
@misc{harsy2026nnfoundation3dfoundationmodels,
  title         = {nnFoundation: 3D Foundation Models for Radiology},
  author        = {Ulrich Harsy, Constantin and Wald, Tassilo and Gotkowski, Karol and others},
  year          = {2026},
  eprint        = {2609.26924},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.26924}
}

@misc{suprijadi2026resolutionmeetsreduction,
  title         = {Resolution Meets Reduction: Efficient Visual Context for 3D Radiology Report Generation},
  author        = {Suprijadi, Jonathan and Stock, Raphael and Langenberg, Moritz
                   and Zimmerer, David and Kahl, Kim-Celine and Denner, Stefan
                   and Kirchhoff, Yannick and Gotkowski, Karol and Rokuss, Maximilian
                   and Traub, Jeremias and Wald, Tassilo and Ulrich, Constantin
                   and Maier-Hein, Klaus},
  year          = {2026},
  eprint        = {2608.08713},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.08713}
}
```

The nnFoundation model cards provide the full citation and additional references
for the vision encoders and pretraining framework.

## License

Project code, documentation, and configurations are licensed under
[Apache 2.0](LICENSE.md). See [NOTICE.md](NOTICE.md) for upstream attribution.
External publications, datasets, pretrained models, and dependencies retain their
respective licenses.
