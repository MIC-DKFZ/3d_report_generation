"""Create exact cosine-neighbor mappings from precomputed embeddings."""

import argparse
from collections import Counter
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Tuple

import blosc2
import faiss
import numpy as np


class PoolingMethod(Enum):
    GAP = "gap"  # Global Average Pooling
    GMP = "gmp"  # Global Max Pooling
    MIX = "mix"  # Concatenation of GAP and GMP
    GEM = "gem"  # Generalized Mean Pooling
    RMAC = "rmac"  # Regional Maximum Activations of Convolutions
    FLATTEN = "flatten"  # No pooling: flatten all non-batch dims for exact retrieval


def aggregate_and_normalize_embedding(
    arr: np.ndarray, method: PoolingMethod = PoolingMethod.GAP, gem_p: float = 3.0
) -> np.ndarray:
    """
    Input shapes:
        - (1, C, W, H, D)  -> pools over (W, H, D)
        - (1, C, L)        -> pools over (L)
        - (1, C)           -> flattened to (C,)
        - (C,)             -> left as is

    Output:
        - (C,) or (2C,) float32, L2-normalized.
    """
    # 0. Add batch dimension if first dim is not == 1
    if arr.shape[0] != 1:
        arr = arr[np.newaxis, :]  # (X) -> (1, X)
    arr = arr.astype(np.float32, copy=False)

    # 1. Identify Spatial Axes based on dimensions
    spatial_axes = None

    if arr.ndim == 5:
        # (1, C, W, H, D)
        spatial_axes = (2, 3, 4)
    elif arr.ndim == 3:
        # (1, C, L)
        spatial_axes = (2,)
    elif arr.ndim == 2:
        # (1, C) -> No spatial dims to pool
        if arr.shape[0] != 1:
            raise ValueError(f"Expected shape (1, C) for 2D array, got {arr.shape}")
        emb = arr[0]
        return _normalize(emb)
    elif arr.ndim == 1:
        # (C,) -> Already pooled
        return _normalize(arr)
    else:
        raise ValueError(f"Unexpected array shape: {arr.shape}")

    # 2. Apply Pooling Strategy

    if method == PoolingMethod.FLATTEN:
        # No spatial aggregation: flatten C + spatial dims into one feature vector.
        # (1, C, ...)->(1, C*...)
        emb = arr.reshape(arr.shape[0], -1)

    elif method == PoolingMethod.GAP:
        emb = arr.mean(axis=spatial_axes)

    elif method == PoolingMethod.GMP:
        emb = arr.max(axis=spatial_axes)

    elif method == PoolingMethod.MIX:
        # Concatenate Average and Max pooling
        # Result is (1, 2C) -> (2C,)
        avg_pool = arr.mean(axis=spatial_axes)
        max_pool = arr.max(axis=spatial_axes)
        emb = np.concatenate([avg_pool, max_pool], axis=1)

    elif method == PoolingMethod.GEM:
        # Generalized Mean: (mean(x^p))^(1/p)
        # Note: We clamp min value to avoid NaNs with fractional powers if input is near 0
        p = gem_p
        arr_clamped = np.maximum(arr, 1e-6)
        emb = np.power(np.mean(np.power(arr_clamped, p), axis=spatial_axes), 1.0 / p)

    elif method == PoolingMethod.RMAC:
        # Simplified R-MAC: Sum of Max-pooled regions + Global Max
        # This implementation uses a fixed 2x2(x2) grid approach for efficiency in NumPy
        emb = _compute_rmac_grid(arr, spatial_axes)

    else:
        raise ValueError(f"Unknown pooling method: {method}")

    # 3. Final Shape Check
    if emb.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {emb.shape}")

    emb = emb[0]  # Flatten batch dim

    # 4. Normalize
    return _normalize(emb)


def _normalize(emb: np.ndarray) -> np.ndarray:
    """Helper to perform L2 normalization safely."""
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def _compute_rmac_grid(arr: np.ndarray, axes: Tuple[int, ...]) -> np.ndarray:
    """
    Computes a simplified R-MAC by splitting the spatial dimensions into a grid,
    max-pooling each grid cell, summing them up, and adding the global max.
    """
    # Start with global max
    final_emb = arr.max(axis=axes)

    # If 5D (3 spatial dims), split into 2x2x2 grid (8 regions)
    if len(axes) == 3:
        _, _, W, H, D = arr.shape
        # Simple non-overlapping windows (or slightly overlapping depending on rounding)
        w_mid, h_mid, d_mid = W // 2, H // 2, D // 2

        # Define 8 corners
        slices = [
            (slice(0, w_mid), slice(0, h_mid), slice(0, d_mid)),
            (slice(w_mid, W), slice(0, h_mid), slice(0, d_mid)),
            (slice(0, w_mid), slice(h_mid, H), slice(0, d_mid)),
            (slice(w_mid, W), slice(h_mid, H), slice(0, d_mid)),
            (slice(0, w_mid), slice(0, h_mid), slice(d_mid, D)),
            (slice(w_mid, W), slice(0, h_mid), slice(d_mid, D)),
            (slice(0, w_mid), slice(h_mid, H), slice(d_mid, D)),
            (slice(w_mid, W), slice(h_mid, H), slice(d_mid, D)),
        ]

        for s_w, s_h, s_d in slices:
            # Check if slice is valid (dimension > 0)
            region = arr[:, :, s_w, s_h, s_d]
            if region.size > 0:
                final_emb += region.max(axis=(2, 3, 4))

    # If 3D (1 spatial dim), split into 2 segments
    elif len(axes) == 1:
        _, _, L = arr.shape
        mid = L // 2
        r1 = arr[:, :, :mid]
        r2 = arr[:, :, mid:]
        if r1.size > 0:
            final_emb += r1.max(axis=2)
        if r2.size > 0:
            final_emb += r2.max(axis=2)

    return final_emb


def read_ids(path):
    records = json.loads(path.read_text())
    ids = [row["image_id"] for row in records]
    if any(not isinstance(image_id, str) or not image_id for image_id in ids):
        raise ValueError(f"Expected nonempty string image IDs in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image IDs in {path}")
    if not ids:
        raise ValueError(f"Dataset JSON contains no samples: {path}")
    return ids


def patient_id(image_id, dataset):
    # MERLIN study IDs do not identify patients, so only exact self is excluded.
    return "_".join(image_id.split("_")[:2]) if dataset == "ctrate" else image_id


def embedding_path(root, image_id, split, dataset, suffix):
    if Path(image_id).name != image_id or image_id in {".", ".."}:
        raise ValueError(f"Image ID must be a filename stem: {image_id!r}")
    directory = root
    if dataset == "ctrate":
        directory = root / ("valid_fixed" if split == "test" else "train_fixed")
    return directory / f"{image_id}{suffix}"


def load_vector(path, method):
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix == ".bl2":
        array = blosc2.load_array(str(path))
    else:
        array = np.asarray(blosc2.open(str(path), mode="r"))
    vector = aggregate_and_normalize_embedding(array, method)
    if not np.isfinite(vector).all() or not np.any(vector):
        raise ValueError(f"Embedding must produce a finite, nonzero vector: {path}")
    return vector


def build_index(ids, root, dataset, suffix, method):
    index = None
    for position, image_id in enumerate(ids):
        path = embedding_path(root, image_id, "train", dataset, suffix)
        vector = load_vector(path, method)
        if index is None:
            index = faiss.IndexFlatIP(len(vector))
        if len(vector) != index.d:
            raise ValueError(f"Inconsistent embedding dimension: {path}")
        index.add(vector.reshape(1, -1))
        if (position + 1) % 1000 == 0:
            logging.info("Indexed %d/%d training embeddings", position + 1, len(ids))
    return index


def retrieve(ids, split, root, dataset, suffix, method, index, train_ids, top_k, batch_size):
    results = {}
    # Search enough extra candidates to replace every excluded same-patient case.
    max_excluded = max(Counter(patient_id(image_id, dataset) for image_id in train_ids).values())
    search_k = min(len(train_ids), top_k + max_excluded)
    index_positions = {image_id: i for i, image_id in enumerate(train_ids)}
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        vectors = []
        for image_id in batch_ids:
            if split == "train" and image_id in index_positions:
                vector = index.reconstruct(index_positions[image_id])
            else:
                path = embedding_path(root, image_id, split, dataset, suffix)
                vector = load_vector(path, method)
                if len(vector) != index.d:
                    raise ValueError(f"Query embedding dimension does not match index: {path}")
            vectors.append(vector)
        similarities, positions = index.search(np.stack(vectors), search_k)
        for image_id, scores, indices in zip(batch_ids, similarities, positions):
            neighbors, neighbor_scores = [], []
            for score, position in zip(scores, indices):
                candidate = train_ids[int(position)]
                if patient_id(candidate, dataset) == patient_id(image_id, dataset):
                    continue
                neighbors.append(candidate)
                neighbor_scores.append(float(score))
                if len(neighbors) == top_k:
                    break
            if not neighbors:
                raise ValueError(f"No eligible training neighbors for query {image_id!r}")
            results[image_id] = {
                "k_nearest_neighbor": neighbors,
                "similarity": neighbor_scores,
            }
        logging.info(
            "Retrieved %s queries: %d/%d", split, min(start + batch_size, len(ids)), len(ids)
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ctrate", "merlin"], required=True)
    parser.add_argument("--emb-root", type=Path, required=True)
    parser.add_argument(
        "--query-emb-root", type=Path, help="Optional separate val/test embedding root."
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        required=True,
        help="Training neighbor pool; use the JSON set as data.train_dataset_json_path.",
    )
    parser.add_argument("--val-json", type=Path)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pooling", choices=[item.value for item in PoolingMethod], default="gap")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-suffix", default="_embedding.bl2")
    args = parser.parse_args()
    if args.top_k < 1 or args.batch_size < 1:
        parser.error("--top-k and --batch-size must be positive")
    inputs = [args.train_json, args.test_json] + ([args.val_json] if args.val_json else [])
    output_names = [
        "retrieval_train.json",
        "retrieval_val.json",
        "retrieval_test.json",
        "retrieval_all.json",
        "metadata.json",
        "train_index.faiss",
    ]
    if {path.resolve() for path in inputs} & {
        (args.output_dir / name).resolve() for name in output_names
    }:
        parser.error("Output files must not overwrite input dataset JSONs")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    method = PoolingMethod(args.pooling)
    splits = {"train": read_ids(args.train_json), "test": read_ids(args.test_json)}
    if args.val_json:
        splits["val"] = read_ids(args.val_json)
    # A combined mapping can have one entry per image ID only.
    for first in splits:
        for second in splits:
            if first < second and set(splits[first]) & set(splits[second]):
                raise ValueError(f"Overlapping query IDs in {first} and {second} JSONs")
    index = build_index(
        splits["train"], args.emb_root, args.dataset, args.embedding_suffix, method
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = {}
    for split, ids in splits.items():
        root = args.emb_root if split == "train" else (args.query_emb_root or args.emb_root)
        mapping = retrieve(
            ids,
            split,
            root,
            args.dataset,
            args.embedding_suffix,
            method,
            index,
            splits["train"],
            args.top_k,
            args.batch_size,
        )
        combined.update(mapping)
        (args.output_dir / f"retrieval_{split}.json").write_text(
            json.dumps(mapping, indent=2) + "\n"
        )
    (args.output_dir / "retrieval_all.json").write_text(json.dumps(combined, indent=2) + "\n")
    faiss.write_index(index, str(args.output_dir / "train_index.faiss"))
    metadata = {
        "train_ids": splits["train"],
        "pooling": args.pooling,
        "dataset": args.dataset,
        "top_k": args.top_k,
        "dimension": index.d,
        "self_exclusion": "patient" if args.dataset == "ctrate" else "study",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
