from typing import Tuple
import nibabel as nib
import nrrd
import yaml
import torch
import os
import numpy as np
import blosc2

import torch.distributed as dist
import os

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