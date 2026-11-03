# RMR embeddings

The RMR configs use precomputed features from Curia, Primus, Merlin, and CT-FM.
Extract them with the corresponding encoder and the preprocessing described in the
[RMR paper](https://arxiv.org/abs/2608.08713) before training. The single-block Primus
features (`Primus-1B_ROI`, `Primus-1B_Whole`) can be generated with
[`generate_embeddings.py`](../vlm/public_scripts/embeddings/generate_embeddings.py), see
[embeddings and paths](data_preparation.md#embeddings-and-paths).

## Folder names and feature shapes

Dataset bases under [`paper_configs/RMR/bases/datasets`](../paper_configs/RMR/bases/datasets/)
set `data.vision_input_path` to `${VLM_EMBEDS}/<DATASET>_<feature name>`, where
`<DATASET>` is `CTRATE` or `MERLIN`. For example, `Curia-8S_ROI.yml` for CT-RATE
reads `${VLM_EMBEDS}/CTRATE_Curia-8S_ROI`.

| Feature name | Meaning | `embed_dim` | `embed_size` (D, H, W) |
| --- | --- | --- | --- |
| `Curia-8S_ROI` / `Curia-8S_Whole` | Curia features pooled to 8 slices, ROI / whole-volume input | 768 | `[8, 32, 32]` |
| `Curia-64S_ROI` | Curia features pooled to 64 slices, ROI input | 768 | `[64, 32, 32]` |
| `Primus-1B_ROI` / `Primus-1B_Whole` | Primus features from one input block (`1B`), ROI / whole-volume input | 864 | `[24, 24, 24]` |
| `Primus-4B_2_1_2_ROI` | Primus features from four input blocks in a 2 × 1 × 2 grid (`4B`, CT-RATE) | 864 | `[48, 24, 48]` |
| `Primus-4B_1_2_2_ROI` | Primus features from four input blocks in a 1 × 2 × 2 grid (`4B`, Merlin) | 864 | `[24, 48, 48]` |
| `Merlin_ROI` | Merlin encoder features, ROI input | 2048 | `[10, 8, 8]` |
| `CTFM_ROI` | CT-FM encoder features, ROI input | 512 | `[12, 12, 12]` |

`ROI` denotes the lung region for CT-RATE and foreground region for Merlin;
`Whole` denotes whole-volume input. Matching the feature shape alone is not
enough to reproduce the experiments: use the same encoder and preprocessing.

Store each prepared array as `[C, D, H, W]` in `<image_id>_embedding.bl2`.
CT-RATE folders contain `train_fixed/` (train/val) and `valid_fixed/` (test);
Merlin files sit directly in the feature folder. See
[embedding paths](data_preparation.md#matching-embedding-folders-to-configs) for
how to arrange folders or adapt the config to your own paths.
