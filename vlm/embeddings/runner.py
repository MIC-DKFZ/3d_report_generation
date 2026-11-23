"""
Embedding a dataset: the volumes of each split folder are preprocessed in DataLoader workers,
encoded on the GPU and saved asynchronously as '<output_dir>/<split folder>/<name>_embedding.bl2'.
Several jobs that differ only in their encoder and output folder (e.g. checkpoints of one
encoder) run in one pass: each volume is preprocessed once and encoded by each of their encoders.
"""

import multiprocessing
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Callable

import blosc2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm.configs.data_config import VisionInputSettings
from vlm.data.preprocessing.nifti_inputs import (
    NiftiEncoderInputDataset,
    collate_samples,
    find_nifti_files,
    strip_nifti_suffix,
)
from vlm.data.vision_input_settings import save_settings
from vlm.embeddings.multiblock import Shape, encode_multiblock, multiblock_sample
from vlm.utils.serialization import save_json


@dataclass
class RuntimeOptions:
    """How the embedding runs; none of these change the embeddings."""

    overwrite: bool = False
    max_files: int | None = None
    num_shards: int = 1
    shard_id: int = 0
    num_workers: int = 4
    prefetch_factor: int = 2
    batch_size: int = 1
    block_batch_size: int = 1
    save_workers: int = 2
    save_threads: int = 8
    cudnn_attention: bool = False
    debug_save_preprocessed: bool = False


@dataclass
class EmbeddingJob:
    """
    What to embed and how. 'settings' holds the orientation and preprocessing (including a
    multi-block grid), saved in the output folder so that inference on new volumes can reproduce
    them. 'input_shape' None takes the encoder's 'patch_size'.
    """

    build_encoder: Callable[[], nn.Module]
    settings: VisionInputSettings
    input_dir: Path
    output_dir: Path
    split_folders: list[str]
    mask_dir: Path | None = None
    input_shape: Shape | None = None
    bf16: bool = False
    options: RuntimeOptions = field(default_factory=RuntimeOptions)

    @property
    def multiblock(self) -> bool:
        return self.settings.vision_input_preprocessing.multiblock


def output_path(output_dir: Path, file: Path) -> Path:
    return output_dir / f"{Path(strip_nifti_suffix(file)).name}_embedding.bl2"


def split_files(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    rank: int = 0,
    world_size: int = 1,
    shard_id: int = 0,
    num_shards: int = 1,
    max_files: int | None = None,
) -> dict[Path, Path]:
    """
    The volumes of a split folder still to embed, mapped to their embedding files: of all its
    (first 'max_files') volumes, every 'num_shards'-th from 'shard_id' on, and of those every
    'world_size'-th from 'rank' on. Independent jobs (shards) and their processes thus split them
    the same way however many each has already saved.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Split folder {input_dir} not found.")
    files = find_nifti_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")
    output_paths = {file: output_path(output_dir, file) for file in files}
    duplicates = len(files) - len(set(output_paths.values()))
    if duplicates:
        raise ValueError(f"{duplicates} volumes in {input_dir} share a file name with another.")
    selected = list(output_paths.items())[:max_files][shard_id::num_shards][rank::world_size]
    return {file: path for file, path in selected if overwrite or not path.exists()}


def save_embedding(embedding: np.ndarray, path: str | Path, nthreads: int = 1) -> None:
    """Save atomically, so that an interrupted run leaves no partial embedding behind."""
    blosc2.set_nthreads(nthreads)
    tmp_path = f"{path}.tmp"
    blosc2.save_array(
        np.ascontiguousarray(embedding),
        tmp_path,
        mode="w",
        cparams={"codec": blosc2.Codec.ZSTD, "clevel": 8},
    )
    os.replace(tmp_path, path)


class AsyncEmbeddingSaver:
    """
    Saves embeddings in a process pool, so that the GPU loop goes on while they are compressed
    (blosc2 holds the GIL, so threads would not help). At most 'max_pending' saves are queued.
    Without workers, embeddings are saved right away.
    """

    def __init__(self, workers: int = 2, nthreads: int = 8, max_pending: int | None = None):
        # Spawned, since forking a process with CUDA initialized is unsafe.
        self.executor = (
            ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
            if workers > 0
            else None
        )
        self.nthreads = nthreads
        self.max_pending = max_pending or 2 * max(workers, 1)
        self.futures = deque()

    def save(self, embedding: torch.Tensor, path: Path) -> None:
        array = embedding.numpy()
        if self.executor is None:
            save_embedding(array, path, self.nthreads)
            return
        while len(self.futures) >= self.max_pending:
            self.futures.popleft().result()
        self.futures.append(self.executor.submit(save_embedding, array, path, self.nthreads))

    def wait(self) -> None:
        while self.futures:
            self.futures.popleft().result()

    def shutdown(self) -> None:
        self.wait()
        if self.executor is not None:
            self.executor.shutdown()


_torch_sdpa = F.scaled_dot_product_attention


def cudnn_sdpa(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw
):
    """
    scaled_dot_product_attention through cuDNN, with the head dim zero-padded to a multiple of 64.
    Head dims that are not a multiple of 8 (e.g. 1056 / 16 = 66 for nnFoundationViT) otherwise
    fall back to FlashAttention-2; cuDNN is ~1.2-1.6x faster on GH200. Zero-padding leaves q·k
    unchanged and the padded output channels are dropped. Calls with masks, dropout or causal
    attention are passed through.
    """
    if attn_mask is not None or dropout_p or is_causal or kw or not query.is_cuda:
        return _torch_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, **kw)
    head_dim = query.shape[-1]
    pad = -head_dim % 64
    q, k, v = (F.pad(t, (0, pad)) for t in (query, key, value))
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = _torch_sdpa(q, k, v, scale=head_dim**-0.5 if scale is None else scale)
    return out[..., :head_dim]


@contextmanager
def attention_backend(cudnn: bool):
    """The encoders look up F.scaled_dot_product_attention at call time (e.g. timm's Eva)."""
    F.scaled_dot_product_attention = cudnn_sdpa if cudnn else _torch_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _torch_sdpa


def init_preprocessing_worker(worker_id: int):
    # DataLoader already limits PyTorch threads; limit SimpleITK's separate pool too.
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def save_preprocessed(inputs: torch.Tensor, directory: Path, name: str, spacing) -> None:
    """Save (n, 1, z, y, x) encoder inputs as NIfTI files, for checking the preprocessing."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, block in enumerate(inputs):
        image = sitk.GetImageFromArray(block[0].float().numpy())
        image.SetSpacing(tuple(spacing or (1.0, 1.0, 1.0))[::-1])
        suffix = "preprocessed" if len(inputs) == 1 else f"block{i}"
        sitk.WriteImage(image, directory / f"{name}_{suffix}.nii.gz")


def encode(encoder: nn.Module, inputs: torch.Tensor, device, dtype, chunk: int) -> torch.Tensor:
    """(n, 1, *input_shape) → (n, C, d, h, w) float32 on the device, 'chunk' inputs at a time."""
    # No torch.cuda.empty_cache() between chunks: reusing the cached attention buffers of large
    # encoders avoids fragmenting the allocator.
    return torch.cat(
        [
            encoder(inputs[i : i + chunk].to(device=device, dtype=dtype)).float()
            for i in range(0, len(inputs), chunk)
        ]
    )


def shared_part(job: EmbeddingJob) -> EmbeddingJob:
    """The job without its encoder and output folder, which jobs run in one pass share."""
    settings = job.settings.model_copy(update={"embedding_encoder": None})
    return replace(job, build_encoder=None, output_dir=None, settings=settings)


def run(*jobs: EmbeddingJob) -> None:
    """Run the jobs in one pass; they may differ only in their encoder and output folder."""
    job = jobs[0]
    if any(shared_part(other) != shared_part(job) for other in jobs[1:]):
        raise ValueError(
            "Jobs run in one pass may differ only in their encoder and output folder."
        )
    if len({other.output_dir for other in jobs}) < len(jobs):
        raise ValueError("Jobs run in one pass need their own output folders.")
    opts = job.options
    # Under torchrun, each process embeds every 'world_size'-th volume of its shard.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    input_dir = job.input_dir.resolve()
    # All splits are listed first, so that a missing one fails before hours of embedding. A
    # volume is embedded if one of the jobs still lacks it, and saved by those that do.
    splits = {}
    for split in job.split_folders:
        todo = [
            split_files(
                input_dir / split,
                other.output_dir / split,
                opts.overwrite,
                rank,
                world_size,
                opts.shard_id,
                opts.num_shards,
                opts.max_files,
            )
            for other in jobs
        ]
        files = [f for f in find_nifti_files(input_dir / split) if any(f in t for t in todo)]
        splits[split] = (files, todo)
    # Before embedding, so that settings differing from the folder's fail early.
    for other in jobs:
        print(f"Settings in {save_settings(other.output_dir, other.settings)}")

    device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if job.bf16 else torch.float32
    encoders = [other.build_encoder().to(device, dtype).eval() for other in jobs]
    input_shapes = {tuple(job.input_shape or encoder.patch_size) for encoder in encoders}
    if len(input_shapes) > 1:
        raise ValueError(f"Jobs run in one pass need one input shape, got {input_shapes}.")
    (input_shape,) = input_shapes
    preprocessing = job.settings.vision_input_preprocessing
    preprocessing_kwargs = preprocessing.preprocessing_kwargs(input_shape)
    preprocess = None
    if job.multiblock:
        preprocess = partial(
            multiblock_sample,
            **preprocessing_kwargs,
            grid=tuple(preprocessing.grid),
            tile_step_size=preprocessing.tile_step_size,
        )

    saver = AsyncEmbeddingSaver(opts.save_workers, opts.save_threads)
    try:
        for split, (files, todo) in splits.items():
            output_dirs = [other.output_dir / split for other in jobs]
            for output_dir in output_dirs:
                output_dir.mkdir(parents=True, exist_ok=True)
            loader = DataLoader(
                NiftiEncoderInputDataset(
                    input_dir / split,
                    files,
                    job.settings.vision_input_orientation,
                    None if job.mask_dir is None else job.mask_dir / split,
                    preprocessing_kwargs,
                    preprocess=preprocess,
                ),
                # Multi-block: one volume per batch, its blocks encoded in chunks.
                batch_size=1 if job.multiblock else opts.batch_size,
                num_workers=opts.num_workers,
                prefetch_factor=opts.prefetch_factor if opts.num_workers > 0 else None,
                collate_fn=collate_samples,
                worker_init_fn=init_preprocessing_worker,
            )
            failed = []
            with torch.inference_mode(), attention_backend(opts.cudnn_attention):
                for samples in tqdm(
                    loader, desc=f"Embedding {split or 'volumes'}", disable=rank != 0
                ):
                    for sample in samples:
                        if sample["error"] is not None:
                            print(f"Skipped {sample['path']}: {sample['error']}")
                            failed.append({key: sample[key] for key in ("path", "error")})
                    samples = [s for s in samples if s["error"] is None]
                    if opts.debug_save_preprocessed:
                        for s in samples:
                            save_preprocessed(
                                s["encoder_input"] if job.multiblock else s["encoder_input"][None],
                                output_dirs[0] / "preprocessed",
                                Path(s["image_id"]).name,
                                preprocessing.target_spacing,
                            )
                    for encoder, output_paths in zip(encoders, todo):
                        needed = [s for s in samples if Path(s["path"]) in output_paths]
                        if not needed:
                            continue
                        if job.multiblock:
                            (sample,) = needed
                            embeddings = encode_multiblock(
                                partial(
                                    encode,
                                    encoder,
                                    device=device,
                                    dtype=dtype,
                                    chunk=opts.block_batch_size,
                                ),
                                sample,
                                input_shape,
                                tuple(preprocessing.grid),
                            )[None]
                        else:
                            inputs = torch.stack([s["encoder_input"] for s in needed])
                            embeddings = encode(encoder, inputs, device, dtype, len(inputs))
                        for s, embedding in zip(needed, embeddings.cpu()):
                            saver.save(embedding, output_paths[Path(s["path"])])
            saver.wait()
            failed_paths = {Path(f["path"]) for f in failed}
            for output_dir, output_paths in zip(output_dirs, todo):
                failed_here = [f for f in failed if Path(f["path"]) in output_paths]
                if failed_here:
                    failed_file = output_dir / (
                        "failed.json"
                        if world_size == 1 and opts.num_shards == 1
                        else f"failed_shard{opts.shard_id}_rank{rank}.json"
                    )
                    save_json(failed_here, failed_file)
                    print(
                        f"Skipped {len(failed_here)} of {len(output_paths)} volumes, "
                        f"see {failed_file}"
                    )
                saved = len(set(output_paths) - failed_paths)
                print(f"Saved {saved} embeddings to {output_dir}")
    finally:
        saver.shutdown()
