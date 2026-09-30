from typing import Tuple, List, Union
from functools import lru_cache
import torch
import os
import numpy as np
import blosc2
from scipy.ndimage import gaussian_filter

import torch.distributed as dist

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(gpu)
        
        return rank, world_size, gpu
    else:
        # Single GPU fallback
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def distribute_files(img_files, rank, world_size):
    """Distribute files across GPUs"""
    # Split files evenly across GPUs
    files_per_gpu = len(img_files) // world_size
    start_idx = rank * files_per_gpu

    if rank == world_size - 1:  # Last GPU gets remaining files
        end_idx = len(img_files)
    else:
        end_idx = start_idx + files_per_gpu

    return img_files[start_idx:end_idx]


def shard_files(img_files, shard_id, num_shards):
    """Select this shard's subset of files for multi-node (multi-job) splitting.

    Each independent job processes a disjoint slice of the (sorted) file list so
    several jobs on different nodes can run in parallel without coordination.
    Uses interleaved (round-robin) assignment so the load stays balanced even if
    the list is ordered by size/case. Within a job the result is further split
    across GPUs by ``distribute_files``.

    Args:
        img_files: Full, deterministically ordered list of files.
        shard_id: Index of this job, in ``[0, num_shards)``.
        num_shards: Total number of jobs the data is split across.

    Returns:
        The subset of ``img_files`` assigned to ``shard_id``.
    """
    if num_shards <= 1:
        return img_files
    assert 0 <= shard_id < num_shards, (
        f"shard_id ({shard_id}) must be in [0, num_shards={num_shards})"
    )
    return img_files[shard_id::num_shards]



def save_embeddings_blosc2(embeddings, output_path, clevel: int = 8, codec=blosc2.Codec.ZSTD):
    blosc2.set_nthreads(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert tensor to numpy if needed
    if hasattr(embeddings, "numpy"):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    print(f"Saving embeddings with shape: {embeddings_np.shape} to {output_path}")

    # Ensure array is contiguous
    embeddings_np = np.ascontiguousarray(embeddings_np)

    cparams = {"codec": codec, "clevel": clevel}

    bytes_saved = blosc2.save_array(embeddings_np, output_path, mode="w", cparams=cparams)
    print(f"Successfully saved {bytes_saved}")
    return


# From nnunetv2.inference.sliding_window_prediction
def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


@lru_cache(maxsize=2)
def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float16, device=torch.device('cuda', 0)) \
        -> torch.Tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
    return gaussian_importance_map
