# Data preparation

## Dataset JSONs

Dataset JSONs define the cases in each training, validation, or test split.
Each record pairs an image ID with a prompt and reference report. The image ID
locates its separately stored vision embeddings; the report provides the training
target and evaluation reference. The scripts below convert the original report
tables into this format, applying the splits and exclusions used in the papers.

Use **v1 for nnFoundation** and **v2 for RMR**. The
[released nnFoundation report-generation models](../README.md#generating-reports-with-the-released-models)
were trained on v2. Both versions can share an output folder because
their filenames include the version. Their differences are listed in the table below.

For [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), the dataset JSON generators expect the original
`radiology_text_reports/train_reports.csv` and
`radiology_text_reports/validation_reports.csv` under the source dataset root.

```bash
uv run vlm/scripts/datasets/ctrate/create_dataset_jsons_v1.py \
  --dataset-path /path/to/CT-RATE/dataset --output-dir /path/to/dataset_jsons/CTRATE
uv run vlm/scripts/datasets/ctrate/create_dataset_jsons_v2.py \
  --dataset-path /path/to/CT-RATE/dataset --output-dir /path/to/dataset_jsons/CTRATE
```

For the [Merlin Abdominal CT Dataset](https://stanford.redivis.com/datasets/exfv-e8h3kvfhs),
the dataset JSON generators expect the original `reports_final.xlsx` spreadsheet.

```bash
uv run vlm/scripts/datasets/merlin/create_dataset_jsons_v1.py \
  --reports-xlsx /path/to/reports_final.xlsx --output-dir /path/to/dataset_jsons/MERLIN
uv run vlm/scripts/datasets/merlin/create_dataset_jsons_v2.py \
  --reports-xlsx /path/to/reports_final.xlsx --output-dir /path/to/dataset_jsons/MERLIN
```

Each command overwrites `<dataset>_<version>_{train,val,test,train_val}.json`.
`train_val` combines train and validation. CT-RATE's source validation CSV is used as the
test set. Merlin uses 1,000 fixed validation studies from its source `val` pool
and adds the remaining source validation studies to training.

| Version | CT-RATE | Merlin |
| --- | --- | --- |
| v1 | Excludes scans identified as brain scans; collapses repeated spaces and removes the space before periods | Excludes three duplicated study IDs (eight rows); removes an initial uppercase `FINDINGS:` header |
| v2 | Same scan exclusions as v1, plus suspected brain scans with lung volume below 1 L measured using TotalSegmentator, and reports with missing findings | Same study exclusions as v1; normalizes line endings and places impression headers on separate lines; removes impression numbering across all splits |

## Embeddings and paths

Training requires precomputed vision embeddings. Generate them with
[`generate_embeddings.py`](../vlm/scripts/embeddings/generate_embeddings.py): the
nnFoundationViT and nnFoundationCNN features of the nnFoundation configs, and the Primus
features of the RMR configs. See [RMR embeddings](rmr_embeddings.md) for all features the
RMR configs expect.

The supplied configs use `${NAME}` placeholders to keep data paths portable.
`VLM_EMBEDS`, `CTRATE_DATASET_JSONS`, and `MERLIN_DATASET_JSONS` are conventions
used by these configs. Set whichever variables your selected config references, or replace their
references in YAML with absolute paths or your own variable names.
The two variables that training requires, `VLM_CONFIG_DIR` and `VLM_OUTPUT_DIR`,
are explained in the [quick start](../README.md#training-quick-start).

```bash
export VLM_EMBEDS="/path/to/embeddings"
export CTRATE_DATASET_JSONS="/path/to/dataset_jsons/CTRATE"
export MERLIN_DATASET_JSONS="/path/to/dataset_jsons/MERLIN"
```

Set each JSON variable to the `--output-dir` used for that dataset's generator: the
folder that contains the `<dataset>_<version>_*.json` files.

Generate the embeddings of one dataset with one command per feature:

```bash
uv run python vlm/scripts/embeddings/generate_embeddings.py \
  --embedding <embedding> --dataset ctrate --input_dir /path/to/CT-RATE/dataset \
  --mask_dir /path/to/CT-RATE/dataset/ts_seg/ts_total --output_base "${VLM_EMBEDS}"
uv run python vlm/scripts/embeddings/generate_embeddings.py \
  --embedding <embedding> --dataset merlin --input_dir /path/to/merlin_data \
  --mask_dir /path/to/totalsegmentator_masks --output_base "${VLM_EMBEDS}"
```

- `--embedding`: the feature to generate, one of the presets in the table below.
- `--dataset`: `ctrate` or `merlin`; selects the dataset's folder layout and region of interest
  (ROI). For CT-RATE, `--input_dir` holds the `train_fixed` and `valid_fixed` folders.
- `--mask_dir`: the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) `total`
  masks, which define the ROI, with the same relative paths as the volumes under `--input_dir`.
  CT-RATE provides them in `ts_seg/ts_total`; for Merlin, make them with TotalSegmentator as
  described in [masks](nifti_inference.md#masks). `Primus-1B_Whole` needs no masks.
- `--output_base`: the embeddings are saved in `${VLM_EMBEDS}/<DATASET>_<embedding>`, where the
  configs expect them.

The encoders are downloaded from Hugging Face (`--checkpoint` takes a local file instead).

| Paper     | `--embedding` | Encoder | Input | CT-RATE | Merlin |
|-----------| --- | --- | --- | --- | --- |
| nnFoundation | `nnFoundationViT` | [nnFoundationViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT) | ROI | Center crop | Resized |
| nnFoundation | `nnFoundationCNN` | [nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN) | ROI | Resized | Resized |
| nnFoundation | `nnFoundationViT_release` | [nnFoundationViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT) | Foreground | Resized directly | Resized directly |
| RMR       | `Primus-1B_ROI` | [PrimusM-MAE](https://huggingface.co/MIC-DKFZ/PrimusM-MAE) | ROI | Resized | Resized |
| RMR       | `Primus-1B_Whole` | [PrimusM-MAE](https://huggingface.co/MIC-DKFZ/PrimusM-MAE) | Whole volume | Resized | Resized |
| RMR       | `Primus-4B_2_1_2_ROI` | [PrimusM-MAE](https://huggingface.co/MIC-DKFZ/PrimusM-MAE) | ROI, 2 × 1 × 2 blocks | Resized | Resized |
| RMR       | `Primus-4B_1_2_2_ROI` | [PrimusM-MAE](https://huggingface.co/MIC-DKFZ/PrimusM-MAE) | ROI, 1 × 2 × 2 blocks | Resized | Resized |

Each volume is reoriented (to the dataset's orientation, LPS for CT-RATE and LAS for Merlin,
unless the encoder or `--orientation` sets another), z-score normalized, resampled to 1 mm, cropped to its ROI (the lungs
for CT-RATE, the foreground for Merlin) and center cropped or resized to the encoder input of
192³, as listed above. The multi-block features (`4B`) resize the ROI to a window of four
encoder inputs, encode them one by one and merge their embeddings into one grid. The script's
docstring describes each step in detail.

`--help` lists the options that change a preset (e.g. `--roi lungs` or `--roi foreground`
instead of the papers' ROI, `--processing_mode`, `--grid`, `--orientation`, `--bf16`) and those for large runs (`--batch_size`, `--num_shards`,
`--shard_id`; launched with `torchrun`, the volumes are split across its processes). Changed
options are appended to the folder name; `--output_dir` instead of `--output_base` sets the
folder itself. Existing embeddings are skipped, so an interrupted run can be restarted.

The settings are saved as `vision_input_preprocessing.yml` in the output folder. Training copies
them into the run config, so that inference on new volumes
([`inference_nifti.py`](../inference_nifti.py)) preprocesses and encodes them the same way.

### Embeddings of the released models

The [released report-generation models](../README.md#generating-reports-with-the-released-models)
use other settings than the paper presets, collected in `nnFoundationViT_release`: on both
datasets, the foreground of all labels as ROI, cropped at the volume's own spacing and resized
once to the encoder input (`resized_direct`), and the nnFoundationViT revision they were trained
with:

```bash
uv run python vlm/scripts/embeddings/generate_embeddings.py \
  --embedding nnFoundationViT_release --dataset ctrate --input_dir /path/to/CT-RATE/dataset \
  --mask_dir /path/to/CT-RATE/dataset/ts_seg/ts_total --output_base "${VLM_EMBEDS}"
```

For Merlin, use `--dataset merlin` with its volumes and masks as above. The saved settings then
match the `data` settings in the released models' `final_config.yml`, which records all their
training settings. Generating reports for new volumes with
[`inference_nifti.py`](nifti_inference.md) needs no embeddings.

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

`generate_embeddings.py` with `--output_base "${VLM_EMBEDS}"` creates exactly these folders.
With `--output_dir` or changed options, rename or symlink the output folder, or edit
`data.vision_input_path` in your dataset base to point to it. Renaming alone does not make features compatible: the
encoder, preprocessing, `embed_dim`, and `embed_size` must match the recipe.

Files are named `<image_id>_embedding.bl2`. CT-RATE uses `train_fixed/` for train/val
and `valid_fixed/` for test; Merlin stores files directly in the encoder folder.
Features have shape `[C, D, H, W]`.
