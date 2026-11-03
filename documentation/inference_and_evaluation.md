# Inference and evaluation

Run from the repository root with the [data paths](data_preparation.md#embeddings-and-paths)
and `VLM_CONFIG_DIR` set for the run's paper. Inference requires its vision embeddings
and trained checkpoint; evaluation of saved predictions requires neither.

## Automatic evaluation after training

The config fields `eval.run_inference` and `eval.run_eval` are `true` by default, so
`train.py` automatically generates reports and evaluates them after training.
The checkpoint and split below are set by the supplied
[`paper_configs/`](../paper_configs/):

| Project | Checkpoint (`eval.mode`) | Split (`eval.split`) |
| --- | --- | --- |
| nnFoundation | `final` | `test` |
| RMR | `best` | `val` |

nnFoundation therefore already produces test results by default. RMR needs a
separate test run, or `--eval.split test` when launching training.

## Output structure

Predictions and metrics are saved inside the selected checkpoint. For example:

```text
${VLM_OUTPUT_DIR}/ctrate/CTRATE_nnFoundationViT/<run>/
├── final_config.yml
└── checkpoint-1000/
    └── inference/
        ├── final_config.yml
        ├── inference_results.json
        ├── VLM_metrics.json
        └── metrics_per_case.csv
```

The run-level `final_config.yml` records training settings; the inference copy
records the settings used to generate reports. Pass the run folder to
`inference.py --run_dir` and the checkpoint's `inference/` folder to
`eval.py --inference_dir`.

## Test evaluation for RMR

Generate and evaluate test reports using the best validation checkpoint:

```bash
uv run python inference.py \
  --run_dir /path/to/run --mode best --eval.split test \
  --suffix _test
```

`inference.py` also runs evaluation when the saved config has
`eval.run_eval: true`, so no separate `eval.py` call is needed. It saves
`inference_results.json` and metrics in the run's inference output directory.
The suffix saves this test run under `checkpoint-<step>/inference_test/`, keeping
the default validation outputs in `inference/`.
To repeat nnFoundation inference, use `--mode final`, with an
`--suffix` to keep the existing results in `inference/`.

## Evaluate saved predictions

Recompute metrics from an existing `inference_results.json`:

```bash
uv run python eval.py \
  --inference_dir /path/to/run/checkpoint-1000/inference --dataset_name ctrate
```

Use `--dataset_name merlin` for Merlin. Records must contain `image_id`,
`output_text_clean`, and `answer`.

Scoring includes BLEU, METEOR, ROUGE-L, CIDEr, and clinical finding metrics using
[RadBERT-CT](https://huggingface.co/IAMJB/RadBERT-CT) for CT-RATE or
[ModernBERT-Merlin](https://huggingface.co/MIC-DKFZ/ModernBERT-Merlin) for Merlin.
Optionally pass `--gt_labels_path /path/to/gt_labels.csv` to override Merlin's
reference labels, which default to `gt_labels.csv` from the ModernBERT-Merlin
repository.
