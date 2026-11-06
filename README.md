# Generating CT reports with the evaluation checkpoints

This archive holds only the code to generate radiology reports from CT volumes (NIfTI) with two
checkpoints of Qwen3-VL-2B trained on nnFoundationViT embeddings:

| Checkpoint | Trained on | Hugging Face repository |
|---|---|---|
| CT-RATE | chest CT (CT-RATE v2) | `<user>/<ctrate repository>` (revision `<hash>`) |
| Merlin | abdominal CT (Merlin) | `<user>/<merlin repository>` (revision `<hash>`) |

Each volume is reoriented, cropped to its TotalSegmentator mask, preprocessed and encoded as
the training volumes were, so that no settings have to be passed. The checkpoints are
preliminary; please do not share them further.

## Installation

Python 3.12 and a CUDA GPU are needed. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra totalsegmentator
```

or with pip in a new environment: `pip install -e ".[totalsegmentator]"`.

The checkpoints were trained with FlashAttention 3. It is optional: without it, the models are
loaded with PyTorch's SDPA attention (with a warning), which gives the same reports up to small
numerical differences.

## Downloads

The checkpoints, the Qwen3-VL base model, the vision encoder and the TotalSegmentator weights
are downloaded on first use. On machines without internet access (e.g. cluster compute nodes),
download them beforehand on a machine with access to the same file system:

```bash
hf download <user>/<ctrate repository> --revision <hash> --local-dir models/ctrate
hf download <user>/<merlin repository> --revision <hash> --local-dir models/merlin
hf download Qwen/Qwen3-VL-2B-Instruct
hf download MIC-DKFZ/nnFoundationViT
totalseg_download_weights -t total
```

and set `HF_HUB_OFFLINE=1` for the runs. On Slurm or LSF clusters, set `HF_HUB_CACHE` to the
Hugging Face cache the downloads went to (outside clusters, `~/.cache/huggingface/hub` is used).
`TOTALSEG_HOME_DIR` sets where TotalSegmentator keeps its weights (default
`~/.totalsegmentator`). TotalSegmentator sends anonymous usage statistics unless
`"send_usage_stats": false` is set in `<TOTALSEG_HOME_DIR>/config.json`.

## Generating reports

```bash
torchrun --standalone --nproc_per_node=<number of GPUs> inference_nifti.py \
    --checkpoint_dir models/ctrate --input_dir <folder of chest CT volumes> \
    --mask_dir <folder for the masks> --output_dir <output folder>
```

and likewise with `models/merlin` for abdominal CT. One GPU works too
(`python inference_nifti.py ...`).

- `--input_dir` is searched recursively for `.nii.gz` and `.nii` files. The report ID of a
  volume is its path relative to `--input_dir`, without the suffix.
- The volumes are cropped to TotalSegmentator `total` masks (v2, foreground of all labels).
  Masks that already exist in `--mask_dir`, with the same relative path as the volume, are
  used; missing ones are made with TotalSegmentator 2.18.0 and saved there, so that later runs
  reuse them. Without `--mask_dir`, they are made in `<output folder>/totalsegmentator_masks`.
  Own masks must be TotalSegmentator v2 `total` segmentations on the grid of their volume.
- TotalSegmentator takes about 2 to 4 minutes per volume and GPU (on an H100-class GPU), much
  longer than the report itself; the volumes are split across the GPUs of a `torchrun` run.
- Volumes that cannot be read, or without a usable mask (missing, empty, or not on the grid of
  the volume), are skipped and listed under `failed` in `inference_nifti_settings.json`. With
  `--allow_missing_masks`, volumes without a mask or with an empty one are used whole instead;
  the models saw almost no whole volumes in training, so these reports are less reliable. They
  are marked with `"mask": "missing, whole volume"`.

## Outputs to send back

From each output folder:

- `inference_results.json`: one report per volume (`output_text_clean`), with whether its mask
  was read, made, or missing (`mask`).
- `inference_nifti_settings.json`: the preprocessing, encoder and mask settings used, and the
  skipped volumes.
- `final_config.yml`: the checkpoint's config as used.

Please also tell us the version of this archive (the contents of `VERSION`).
