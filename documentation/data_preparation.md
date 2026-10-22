# Data preparation

## Dataset JSONs

Dataset JSONs define the cases in each training, validation, or test split.
Each record pairs an image ID with a prompt and reference report. The image ID
locates its separately stored vision embeddings; the report provides the training
target and evaluation reference. The scripts below convert the original report
tables into this format, applying the splits and exclusions used in the papers.

Use **v1 for nnFoundation** and **v2 for RMR and MIRAGE**. Both versions can share an output
folder because their filenames include the version. Their differences are listed
in the table below.

For [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), the dataset JSON generators expect the original
`radiology_text_reports/train_reports.csv` and
`radiology_text_reports/validation_reports.csv` under the source dataset root.

```bash
uv run vlm/public_scripts/datasets/ctrate/create_dataset_jsons_v1.py \
  --dataset-path /path/to/CT-RATE/dataset --output-dir /path/to/dataset_jsons/CTRATE
uv run vlm/public_scripts/datasets/ctrate/create_dataset_jsons_v2.py \
  --dataset-path /path/to/CT-RATE/dataset --output-dir /path/to/dataset_jsons/CTRATE
```

For the [Merlin Abdominal CT Dataset](https://stanford.redivis.com/datasets/exfv-e8h3kvfhs),
the dataset JSON generators expect the original `reports_final.xlsx` spreadsheet.

```bash
uv run vlm/public_scripts/datasets/merlin/create_dataset_jsons_v1.py \
  --reports-xlsx /path/to/reports_final.xlsx --output-dir /path/to/dataset_jsons/MERLIN
uv run vlm/public_scripts/datasets/merlin/create_dataset_jsons_v2.py \
  --reports-xlsx /path/to/reports_final.xlsx --output-dir /path/to/dataset_jsons/MERLIN
```

Each command overwrites `<dataset>_<version>_{train,val,test,train_val}.json`.
`train_val` combines train and validation. CT-RATE's source validation CSV is used as the
test set. Merlin uses 1,000 fixed validation studies from its source `val` pool
and adds the remaining source validation studies to training.

| Version | CT-RATE | Merlin |
| --- | --- | --- |
| v1 | Excludes scans identified as brain scans; retains impression-only reports and `Not given.` impressions; collapses repeated spaces and removes the space before periods | Excludes three duplicated study IDs (eight rows); removes an initial uppercase `FINDINGS:` header; preserves other headers, impression numbering, and line endings |
| v2 | Same scan exclusions as v1, plus suspected brain scans with lung volume below 1 L measured using TotalSegmentator, and reports with missing findings | Same study exclusions as v1; normalizes line endings and places impression headers on separate lines; removes impression numbering across all splits |

## Embeddings and paths

Training requires precomputed vision embeddings. Generate nnFoundationViT and
nnFoundationCNN features with
[`generate_nnfoundation_embeddings.py`](../vlm/public_scripts/embeddings/generate_nnfoundation_embeddings.py);
see [RMR embeddings](rmr_embeddings.md) for the features the RMR configs expect.

The supplied configs use `${NAME}` placeholders to keep data paths portable.
`VLM_EMBEDS`, `CTRATE_DATASET_JSONS`, and `MERLIN_DATASET_JSONS` are conventions
used by these configs; `VLM_RETRIEVAL` serves the same purpose for retrieval.
Set whichever variables your selected config references, or replace their
references in YAML with absolute paths or your own variable names.
The framework's two required variables, `VLM_CONFIG_DIR` and `VLM_OUTPUT_DIR`,
are explained in the [quick start](../README.md#quick-start).

```bash
export VLM_EMBEDS="/path/to/embeddings"
export CTRATE_DATASET_JSONS="/path/to/dataset_jsons/CTRATE"
export MERLIN_DATASET_JSONS="/path/to/dataset_jsons/MERLIN"
```

The JSON variables point directly to the generator output folders.

### Matching embedding folders to configs

`VLM_EMBEDS` points to the parent directory of your embedding folders. Their names
come from `data.vision_input_path` in the selected dataset base, for example
[`nnFoundationViT.yml`](../paper_configs/nnFoundation/bases/datasets/ctrate/nnFoundationViT.yml):

```yaml
data:
  vision_input_path: ${VLM_EMBEDS}/CTRATE_nnFoundationViT
  embed_dim: 1056
  embed_size: [24, 24, 24]
```

With `VLM_EMBEDS=/path/to/embeddings`, this expects:

```text
/path/to/embeddings/
└── CTRATE_nnFoundationViT/
    ├── train_fixed/<image_id>_embedding.bl2
    └── valid_fixed/<image_id>_embedding.bl2
```

These are config naming conventions, not names automatically assigned by training.
Place your generated features at the configured path (rename or symlink the output
folder if needed), or edit `data.vision_input_path` in your dataset base to point
to their actual location. Renaming alone does not make features compatible: the
encoder, preprocessing, `embed_dim`, and `embed_size` must match the recipe.

Files are named `<image_id>_embedding.bl2`. CT-RATE uses `train_fixed/` for train/val
and `valid_fixed/` for test; Merlin stores files directly in the encoder folder.
Features have shape `[C, D, H, W]`.

For retrieval experiments, continue with [retrieval mappings](retrieval.md).
