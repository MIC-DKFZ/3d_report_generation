# Retrieval mappings

Run [`create_retrieval_files.py`](../vlm/public_scripts/retrieval/create_retrieval_files.py)
on existing vision embeddings and [dataset JSONs](data_preparation.md).
The script pools features, L2-normalizes them, and finds exact cosine neighbors
with FAISS. It does not encode images or reports.

```bash
export VLM_RETRIEVAL="/path/to/retrieval"

uv run vlm/public_scripts/retrieval/create_retrieval_files.py \
  --dataset ctrate \
  --emb-root /path/to/CTRATE_nnFoundationViT \
  --train-json /path/to/dataset_jsons/CTRATE/ctrate_v2_train.json \
  --test-json /path/to/dataset_jsons/CTRATE/ctrate_v2_test.json \
  --pooling gap --top-k 100 \
  --output-dir "${VLM_RETRIEVAL}/CTRATE/nnFoundationViT"
```

Use the same training JSON as `data.train_dataset_json_path`: it defines the
neighbor pool. Sample limits such as `data.n_train_samples` limit queries only;
the full training JSON remains the reference pool, and neighbors absent from it
are filtered out. An optional `--val-json` adds validation queries.
`--query-emb-root` reads validation/test query embeddings from a different root
than the training pool.

Use the [embedding filenames and folder layout](data_preparation.md#embeddings-and-paths)
described in data preparation.

CT-RATE excludes every candidate from the query's patient; Merlin excludes the
query study. Missing or invalid embeddings stop generation. Smaller eligible
pools return fewer than `top_k` neighbors; an empty pool is an error.

Outputs are `retrieval_all.json`, per-split JSONs, `train_index.faiss`, and
`metadata.json`. Mapping entries contain `k_nearest_neighbor` and `similarity`
lists. Add `--help` to inspect pooling and other options.

## Use the mapping

Set the dataset config to the generated file:

```yaml
data:
  retrieval_json_path: ${VLM_RETRIEVAL}/CTRATE/nnFoundationViT/retrieval_all.json
  retrieval_n_samples: 3
  retrieval_use_image_in_prompt: false
```

Generate one mapping per encoder under `${VLM_RETRIEVAL}/<dataset>/<encoder>/`,
using dataset names `CTRATE` or `MERLIN`. The same ranked mapping supports different
neighbor counts and report-only or image/report retrieval; image retrieval loads
the neighbors' existing vision embeddings. No separate text embeddings or
reranking step is required. To disable retrieval, set `retrieval_json_path: null`.

## MIRAGE recipes

The [MIRAGE configs](../paper_configs/MIRAGE/) train Qwen3-VL-2B with a QwenAdapter
on CT-RATE v2 dataset JSONs. Recipes are named
`Qwen3-VL-2B__<encoder>__QwenAdapter__<variant>.yml`, with variants:

| Variant | Retrieval |
| --- | --- |
| `NoRetrieval` | Baseline; no mapping is loaded |
| `Txt_K1`, `Txt_K3`, `Txt_K10` | Adds the reports of 1, 3, or 10 neighbors to the query image |
| `ImgTxt_K1` | Adds one neighbor's image and report |

| Encoder | Embeddings | Retrieval mapping |
| --- | --- | --- |
| `nnFoundationViT` | `${VLM_EMBEDS}/CTRATE_nnFoundationViT` | `${VLM_RETRIEVAL}/CTRATE/nnFoundationViT/retrieval_all.json` |
| `Curia-8S_ROI` | `${VLM_EMBEDS}/CTRATE_Curia-8S_ROI` | `${VLM_RETRIEVAL}/CTRATE/Curia-8S_ROI/retrieval_all.json` |

Generate each mapping as above from `ctrate_v2_train.json` and `ctrate_v2_test.json`.
Recipes already include their dataset and retrieval bases, so no `--add_base_configs`
is needed:

```bash
export VLM_CONFIG_DIR="$PWD/paper_configs/MIRAGE"

uv run python train.py \
  --train_config Qwen3-VL-2B__nnFoundationViT__QwenAdapter__ImgTxt_K1.yml
```

MIRAGE disables validation during training and evaluates the final checkpoint on test.
