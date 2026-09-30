from typing import Tuple, List, Union
from functools import lru_cache
import nibabel as nib
import nrrd
import yaml
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



def load_yaml(file_path: str):
    try:
        with open(file_path, "r") as f:
            rocket_data = yaml.safe_load(f)
            return rocket_data
    except FileNotFoundError:
        raise f"Rocket data file not found at {file_path}"
    except yaml.YAMLError as e:
        raise f"Error parsing Rocket data file: {str(e)}"


# --- Image Processing Functions ---
def load_nrrd(img_path):
    """
    Loads a NRRD image (.nrrd) and returns data and affine matrix.

    Args:
        img_path (str): Path to the NRRD image.

    Returns:
        tuple: (img_data, affine)
    """
    data, header = nrrd.read(img_path)
    img_data = np.array(data)  # Ensure it's a NumPy array

    # Construct the affine matrix from NRRD header information
    space_directions = header['space directions']
    space_origin = header['space origin']

    # Handle cases where 'space directions' might have 'none' entries (e.g., for scalar data)
    valid_directions = []
    for direction in space_directions:
        if not np.isnan(direction).any(): #or not any(d is None for d in direction) #removed this as it does raise a type error.
            valid_directions.append(direction)
    valid_directions = np.array(valid_directions)

    affine = np.eye(4)
    affine[:valid_directions.shape[0], :valid_directions.shape[1]] = valid_directions
    affine[:3, 3] = space_origin
    #The following line is needed as otherwise some files do have a shape which is not correct for multiplication
    affine = affine.T
    return img_data, affine


def load_nifti(img_path):
    """
    Loads a NIfTI image (.nii or .nii.gz) and returns data and affine matrix.

    Args:
        img_path (str): Path to the NIfTI image.

    Returns:
        tuple: (img_data, affine)
    """
    img = nib.load(img_path)
    img_data = img.get_fdata()
    affine = img.affine
    return img_data, affine


def load_image_data_and_affine(img_path: str):
    """
    Loads image data and affine matrix based on file extension.

    Args:
        img_path (str): Path to the image file.

    Returns:
        tuple: (img_data, affine) where:
            img_data (np.ndarray): The image data as a NumPy array.
            affine (np.ndarray): The affine transformation matrix.

    Raises:
        ValueError: If the file extension is not supported.
    """
    _, ext = os.path.splitext(img_path)  # Get file extension (e.g., '.nii.gz')
    if ext == ".gz":
        _, ext_inner = os.path.splitext(img_path[:-3]) # Remove .gz and get next extension
        ext = ext_inner + ext


    if ext == ".nii.gz" or ext == ".nii":
        return load_nifti(img_path)
    elif ext == ".nrrd":
        return load_nrrd(img_path)
    elif ext == ".b2nd":
        return load_blosc2(img_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def load_blosc2(img_path: str) -> Tuple[np.ndarray, np.ndarray]:
    # von nnunet kopiert...
    dparams = {
        'nthreads': 1
    }

    mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}

    data_array = blosc2.open(urlpath=img_path, mode='r', dparams=dparams, **mmap_kwargs)
    print(f"blosc2 array loaded, dtype: {data_array.dtype}, shape: {data_array.shape}")

    return data_array, None


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