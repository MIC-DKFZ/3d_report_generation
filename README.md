# Generating CT reports with the evaluation checkpoints

This archive holds only the code to generate radiology reports from CT volumes (NIfTI) with two
checkpoints of Qwen3-VL-2B trained on nnFoundationViT embeddings:

| Checkpoint | Trained on | Hugging Face repository |
|---|---|---|
| CT-RATE | chest CT (CT-RATE v2) | [`jonathan-suprijadi/ct-report-ctrate`](https://huggingface.co/jonathan-suprijadi/ct-report-ctrate) (revision `51362130`) |
| Merlin | abdominal CT (Merlin) | [`jonathan-suprijadi/ct-report-merlin`](https://huggingface.co/jonathan-suprijadi/ct-report-merlin) (revision `9cdb8cf5`) |

Each volume is reoriented, cropped to its TotalSegmentator mask, preprocessed and encoded as
the training volumes were, so that no settings have to be passed.

## Installation

Python 3.12 and a CUDA GPU are needed. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra totalsegmentator
```

or with pip in a new environment: `pip install -e ".[totalsegmentator]"`.

FlashAttention 3 is not needed; the warning about it when loading can be ignored.

## Downloads

Download the two checkpoints into `models/` (`--checkpoint_dir` takes a local folder):

```bash
hf download jonathan-suprijadi/ct-report-ctrate --revision 5136213031163fd34cceabd15362efdffc11c9b4 --local-dir models/ctrate
hf download jonathan-suprijadi/ct-report-merlin --revision 9cdb8cf5ef03c747b702f6a58550041c3f749ecd --local-dir models/merlin
```

The Qwen3-VL base model, the vision encoder and the TotalSegmentator weights are downloaded on
first use. On machines without internet access (e.g. cluster compute nodes), download them
beforehand on a machine with access to the same file system:

```bash
hf download Qwen/Qwen3-VL-2B-Instruct
hf download MIC-DKFZ/nnFoundationViT
totalseg_download_weights -t total
```

and set `HF_HUB_OFFLINE=1` for the runs. On Slurm or LSF clusters, set `HF_HUB_CACHE` to the
Hugging Face cache the downloads went to (outside clusters, `~/.cache/huggingface/hub` is used).
TotalSegmentator keeps its weights in `TOTALSEG_HOME_DIR` (default `~/.totalsegmentator`) and
sends anonymous usage statistics unless `"send_usage_stats": false` is set in `config.json` there.

## Generating reports

```bash
torchrun --standalone --nproc_per_node=<number of GPUs> inference_nifti.py \
    --checkpoint_dir models/ctrate --input_dir <folder of chest CT volumes> \
    --mask_dir <folder for the masks> --output_dir <output folder>
```

and likewise with `models/merlin` for abdominal CT. One GPU works too
(`python inference_nifti.py ...`). `--batch_size <n>` generates `n` reports at once per GPU
(default 1); for batches above 1, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, without
which the GPU memory fragments and runs out.

- `--input_dir` is searched recursively for `.nii.gz` and `.nii` files. The report ID of a
  volume is its path relative to `--input_dir`, without the suffix.
- The volumes are cropped to TotalSegmentator `total` masks (v2, foreground of all labels).
  Masks that already exist in `--mask_dir`, with the same relative path as the volume, are
  used; missing ones are made with TotalSegmentator 2.18.0 and saved there, so that later runs
  reuse them. Without `--mask_dir`, they are made in `<output folder>/totalsegmentator_masks`.
  Own masks must be TotalSegmentator v2 `total` segmentations on the grid of their volume.
- With many volumes, it is faster to make the masks beforehand with several TotalSegmentator
  processes per GPU (as many as GPU and CPU memory allow), each on its own volumes, and to pass
  them with `--mask_dir`. These are the settings of `inference_nifti.py`:

  ```bash
  TotalSegmentator -i <input_dir>/<volume>.nii.gz -o <mask_dir>/<volume>.nii.gz --ml --task total
  ```

  Masks still missing then (e.g. where a process failed) are made by `inference_nifti.py`.
- Volumes that cannot be read, or without a usable mask (missing, empty, or not on the grid of
  the volume), are skipped and listed under `failed` in `inference_nifti_settings.json`. With
  `--allow_missing_masks`, volumes without a mask or with an empty one are used whole instead;
  the models saw almost no whole volumes in training, so these reports are less reliable. They
  are marked with `"mask": "missing, whole volume"`.

## Outputs

Each output folder holds:

- `inference_results.json`: one report per volume (`output_text_clean`), with whether its mask
  was read, made, or missing (`mask`).
- `inference_nifti_settings.json`: the preprocessing, encoder and mask settings used, and the
  skipped volumes.
- `final_config.yml`: the checkpoint's config as used.
