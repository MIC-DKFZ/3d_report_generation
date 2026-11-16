# Report generation for NIfTI volumes

[`inference_nifti.py`](../inference_nifti.py) generates reports for any folder of NIfTI volumes,
with a released model or a run you trained. Unlike [`inference.py`](inference_and_evaluation.md),
it needs no dataset JSONs or precomputed embeddings: each volume is reoriented, cropped to its
ROI mask, preprocessed and encoded as the training volumes were, then passed to the model.

Run the commands from the repository root.

## Released models

| Model | Trained on | Use for |
| --- | --- | --- |
| [nnFoundationViT-ReportGen-CT-RATE](https://huggingface.co/MIC-DKFZ/nnFoundationViT-ReportGen-CT-RATE) | CT-RATE | Chest CT |
| [nnFoundationViT-ReportGen-Merlin](https://huggingface.co/MIC-DKFZ/nnFoundationViT-ReportGen-Merlin) | Merlin | Abdominal CT |

## Installation and downloads

Install the TotalSegmentator extra, which the masks need, and download the models into a local
folder:

```bash
uv sync --locked --python 3.12 --extra totalsegmentator

uv run hf download MIC-DKFZ/nnFoundationViT-ReportGen-CT-RATE --local-dir models/ctrate
uv run hf download MIC-DKFZ/nnFoundationViT-ReportGen-Merlin --local-dir models/merlin
```

The Qwen3-VL base model, the nnFoundationViT encoder, and the TotalSegmentator weights are
downloaded on first use. On machines without internet access, such as cluster compute nodes,
download them beforehand on a machine that shares the file system:

```bash
uv run hf download Qwen/Qwen3-VL-2B-Instruct
uv run hf download MIC-DKFZ/nnFoundationViT
uv run totalseg_download_weights -t total
```

Then set `HF_HUB_OFFLINE=1` for the runs, and `HF_HUB_CACHE` if the downloads went to a cache
other than the default `~/.cache/huggingface/hub`. TotalSegmentator keeps its weights in
`TOTALSEG_HOME_DIR` (default `~/.totalsegmentator`) and sends anonymous usage statistics unless
`"send_usage_stats": false` is set in `config.json` there.

A warning that FlashAttention 3 is unavailable when loading the model can be ignored.

## Generating reports

```bash
uv run torchrun --standalone --nproc_per_node=<number of GPUs> inference_nifti.py \
  --checkpoint_dir models/ctrate \
  --input_dir /path/to/chest_ct_volumes \
  --mask_dir /path/to/masks \
  --output_dir /path/to/reports
```

Use `models/merlin` for abdominal CT. With `torchrun`, the volumes are split across the GPUs;
`uv run python inference_nifti.py ...` runs on a single GPU.

- `--input_dir` is searched recursively for `.nii.gz` and `.nii` files. The report ID of a
  volume is its path relative to `--input_dir`, without the suffix.
- `--output_dir` defaults to `<checkpoint_dir>/inference_nifti_<input folder name>`.
- `--batch_size <n>` generates `n` reports at once per GPU (default 1). For batches above 1,
  set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid running out of GPU memory
  through fragmentation.

### Masks

The volumes are cropped to TotalSegmentator v2 `total` masks (the foreground of all labels).
Masks in `--mask_dir` with the same relative path as their volume are used; missing ones are
made with TotalSegmentator and saved there, so that later runs reuse them. Without
`--mask_dir`, the masks are made in `<output_dir>/totalsegmentator_masks`. Masks you provide
must be TotalSegmentator v2 `total` segmentations on the grid of their volume.

For many volumes, it is faster to make the masks beforehand, with several TotalSegmentator
processes per GPU (as many as memory allows), each on its own volumes, and the settings
`inference_nifti.py` uses:

```bash
uv run TotalSegmentator -i <input_dir>/<volume>.nii.gz -o <mask_dir>/<volume>.nii.gz --ml --task total
```

Then pass the folder as `--mask_dir`; masks still missing are made by `inference_nifti.py`.
With `--no_generate_masks`, volumes without a mask are skipped instead.

Volumes that cannot be read or have no usable mask (missing, empty, or not on the grid of the
volume) are skipped and listed under `failed` in `inference_nifti_settings.json`. With
`--allow_missing_masks`, volumes without a mask or with an empty one are used whole instead.
The models saw almost no whole volumes in training, so these reports are less reliable; they are
marked with `"mask": "missing, whole volume"`.

## Outputs

Each output folder holds:

- `inference_results.json`: one report per volume (`output_text_clean`), and whether its mask was
  read, made, or missing (`mask`).
- `inference_nifti_settings.json`: the preprocessing, encoder, and mask settings used, and the
  skipped volumes.
- `final_config.yml`: the model's config as used.

## Runs you trained

Pass `--run_dir` (a run folder, or a unique run folder name under `VLM_OUTPUT_DIR`) with
`--mode best` or `final` instead of `--checkpoint_dir`. The preprocessing comes from
`data.vision_input_preprocessing` of the run's config, which training copies from the
`vision_input_preprocessing.yml` of the embedding folder (see
[data preparation](data_preparation.md)). Runs trained on precomputed embeddings encode the
volumes with the encoder recorded in `data.embedding_encoder`.

Runs without these settings need the preprocessing as arguments, at least `--processing_mode`,
and the encoder with `--vision_encoder` and `--vision_encoder_checkpoint`. Masks other than
TotalSegmentator `total` masks are not made by the script and must be passed with `--mask_dir`.
Run `uv run python inference_nifti.py --help` for all options.
