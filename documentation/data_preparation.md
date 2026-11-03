# Data preparation

## Dataset JSONs

Dataset JSONs define the cases in each training, validation, or test split.
Each record pairs an image ID with a prompt and reference report. The image ID
locates its separately stored vision embeddings; the report provides the training
target and evaluation reference. The scripts below convert the original report
tables into this format, applying the splits and exclusions used in the papers.

Use **v1 for nnFoundation** and **v2 for RMR**. Both versions can share an output
folder because their filenames include the version. Their differences are listed
in the table below.

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
| v1 | Excludes scans identified as brain scans; retains impression-only reports and `Not given.` impressions; collapses repeated spaces and removes the space before periods | Excludes three duplicated study IDs (eight rows); removes an initial uppercase `FINDINGS:` header; preserves other headers, impression numbering, and line endings |
| v2 | Same scan exclusions as v1, plus suspected brain scans with lung volume below 1 L measured using TotalSegmentator, and reports with missing findings | Same study exclusions as v1; normalizes line endings and places impression headers on separate lines; removes impression numbering across all splits |

## Embeddings and paths

Training requires precomputed vision embeddings. Generate them with
[`generate_embeddings.py`](../vlm/scripts/embeddings/generate_embeddings.py): the
nnFoundationViT and nnFoundationCNN features of the nnFoundation configs, and the single-block
Primus features of the RMR configs. See [RMR embeddings](rmr_embeddings.md) for all features the
RMR configs expect.

`--embedding` names the features as the configs do (`${VLM_EMBEDS}/<DATASET>_<embedding>`), and
`--dataset` selects the dataset's layout and region of interest (ROI), including CT-RATE's split
folders (`train_fixed` and `valid_fixed`, embedded into the same folders of the output). The ROI
features need the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) `total` masks
of the volumes, stored with the same relative paths: CT-RATE provides them in `ts_seg/ts_total`;
for Merlin, run TotalSegmentator on the volumes. The encoders are downloaded from Hugging Face
(`--checkpoint` takes a local file instead).

```bash
uv run python vlm/scripts/embeddings/generate_embeddings.py \
  --embedding nnFoundationViT --dataset ctrate --input_dir /path/to/CT-RATE/dataset \
  --mask_dir /path/to/CT-RATE/dataset/ts_seg/ts_total \
  --output_dir "${VLM_EMBEDS}/CTRATE_nnFoundationViT"
uv run python vlm/scripts/embeddings/generate_embeddings.py \
  --embedding nnFoundationViT --dataset merlin \
  --input_dir /path/to/merlin_data --mask_dir /path/to/totalsegmentator_masks \
  --output_dir "${VLM_EMBEDS}/MERLIN_nnFoundationViT"
uv run python vlm/scripts/embeddings/generate_embeddings.py \
  --embedding Primus-1B_Whole --dataset ctrate --input_dir /path/to/CT-RATE/dataset \
  --output_dir "${VLM_EMBEDS}/CTRATE_Primus-1B_Whole"
```

| Paper | `--embedding` | Encoder | Input | CT-RATE | Merlin |
| --- | --- | --- | --- | --- | --- |
| nnFoundation | `nnFoundationViT` | [nnFoundationViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT) | ROI | Center crop | Resized |
| nnFoundation | `nnFoundationCNN` | [nnFoundationCNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN) | ROI | Resized | Resized |
| RMR | `Primus-1B_ROI` | [PrimusM-MAE](https://huggingface.co/MIC-DKFZ/PrimusM-MAE) | ROI | Resized | Resized |
| RMR | `Primus-1B_Whole` | [PrimusM-MAE](https://huggingface.co/MIC-DKFZ/PrimusM-MAE) | Whole volume | Resized | Resized |

The ROI is the lungs (labels 10–14) for CT-RATE and the foreground of all labels, without
components below 100 voxels, for Merlin. All volumes are reoriented (CT-RATE: LPS, Merlin: LAS,
the orientations the datasets are stored in), z-score normalized and resampled to 1 mm. The ROI
is the mask's bounding box, enlarged by 4%; for Merlin, it is further enlarged to at least
192 mm per axis where the volume allows it. The input is then fitted to the encoder input of
192³: a center crop or pad keeps the 1 mm spacing, while resizing gives a spacing of the input's
size divided by 192, per axis. Volumes without a mask are embedded whole, with a warning. The
encoders run in float32 on a GPU if one is available; existing embeddings are skipped, so an
interrupted run can be restarted.

The settings are saved as `vision_input_preprocessing.yml` in the output folder. Training copies
them into the run config, so that inference on new volumes
([`inference_nifti.py`](../inference_nifti.py)) preprocesses and encodes them the same way.

The supplied configs use `${NAME}` placeholders to keep data paths portable.
`VLM_EMBEDS`, `CTRATE_DATASET_JSONS`, and `MERLIN_DATASET_JSONS` are conventions
used by these configs. Set whichever variables your selected config references, or replace their
references in YAML with absolute paths or your own variable names.
The framework's two required variables, `VLM_CONFIG_DIR` and `VLM_OUTPUT_DIR`,
are explained in the [quick start](../README.md#quick-start).

```bash
export VLM_EMBEDS="/path/to/embeddings"
export CTRATE_DATASET_JSONS="/path/to/dataset_jsons/CTRATE"
export MERLIN_DATASET_JSONS="/path/to/dataset_jsons/MERLIN"
```

Set each JSON variable to the `--output-dir` used for that dataset's generator: the
folder that contains the `<dataset>_<version>_*.json` files.

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
