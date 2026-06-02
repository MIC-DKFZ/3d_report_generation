import os
import argparse
from typing import List, Optional, Union, Dict, Tuple, Callable
from pathlib import Path
from tqdm import tqdm
import warnings
from collections import defaultdict

import yaml
import json
import nibabel as nib
import nrrd

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange

# from evavit import PretrainedEvaEncFromEvaMAE, get_config_from_plans

# Assuming nnssl is in the python path or installed
# from nnssl.preprocessing.normalization.normalization_schemes import apply_normalization, NormalizationScheme
from prep.normalization_schemes import (
    apply_normalization,
    NormalizationScheme,
)


class ImagePatchDataset(Dataset):
    def __init__(self,
                 image_paths: List[str],
                 config: dict,
                 vit_patch_size: List[int],
                 processing_function: str,
                 # Add args needed for specific modes
                 segmentation_dir: Optional[str] = None,
                 dataset_json_path: Optional[str] = None, # Or preloaded labels
                 rocket_log_path: Optional[str] = None,   # Or preloaded mapping
                 set_of_interesting_masks: Optional[dict] = None):

        self.image_paths = image_paths
        self.config = config
        self.patch_size = tuple(config['recommended_downstream_patchsize'])
        self.vit_patch_size = vit_patch_size
        self.processing_function = processing_function
        self.segmentation_dir = segmentation_dir
        self.dataset_json_path = dataset_json_path # Consider preloading labels
        self.set_of_interesting_masks = set_of_interesting_masks # Needs careful handling if defined dynamically

        # Pre-load mappings if doing region processing for efficiency
        self.raw2rocket = None
        if self.processing_function == 'region':
            if not rocket_log_path or not segmentation_dir: # Add dataset_json check if strictly needed
                 raise ValueError("Region processing requires rocket_log_path and segmentation_dir")
            _, self.raw2rocket = _create_name_mapping_rocket_raw(rocket_log_path)
            # Load default masks if needed (consider passing the dict directly)
            if self.set_of_interesting_masks is None:
                 from prep.helper import first_group_attempt # Example
                 self.set_of_interesting_masks = {k: v for k, v in first_group_attempt.items() if v != 0}


    def __len__(self):
        # This dataset iterates through IMAGE FILES, not patches
        return len(self.image_paths)

    def _preprocess_and_patch(self, img_data, affine, original_shape, foreground_mask=None, device=torch.device('cpu')):
        """Helper to normalize, tensorize, and patch an image (or cropped region)."""
        img_data_bc = np.expand_dims(img_data, axis=(0, 1))
        normalized_np = apply_normalization(
            scheme=NormalizationScheme.Z_SCORE, # Use config if available
            image=img_data_bc,
            target_dtype=np.float32,
            use_mask_for_norm=False, #(foreground_mask is not None) # Optional: normalize based on mask
        )
        img_tensor = torch.from_numpy(normalized_np).to(device) # Keep on CPU for workers
        fg_tensor = torch.from_numpy(foreground_mask).to(device) if foreground_mask is not None else None

        patches, attn_masks, num_patches_dims = split_image_into_patches(
            img_tensor, self.patch_size, self.vit_patch_size,
            foreground_mask=fg_tensor
        )
        # patches: (N, B=1, C, pH, pW, pD) -> squeeze B
        # attn_masks: (N, aH, aW, aD) -> flatten
        patches = patches.squeeze(1) # Remove B dim -> (N, C, pH, pW, pD)
        n, h, w, d = attn_masks.shape
        # attn_masks_flat = rearrange(attn_masks, 'n h w d -> n (h w d)') # (N, num_vit_tokens)

        return patches, attn_masks, original_shape, affine, num_patches_dims, [h, w, d]


    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image_id = Path(img_path).stem.replace('_0000', '') # Unique ID for this image

        try:
            img_data, affine = load_image_data_and_affine(img_path)
            original_full_shape = img_data.shape

            results = [] # List to hold (patch, mask, metadata) tuples for this image/its regions

            if self.processing_function == 'native':
                patches, attn_masks_flat, _, _, num_patches, num_embeddings_per_dim = self._preprocess_and_patch(img_data, affine, original_full_shape)
                total_patches = patches.shape[0]
                for i in range(total_patches):
                    metadata = {
                        'image_id': image_id,
                        'region_key': 'full', # Indicate it's the full image
                        'patch_index': i,
                        'total_patches': total_patches,
                        'original_shape': original_full_shape, # Shape of the *input* to patching
                        'affine': affine,
                        'num_patches_dim': num_patches,
                        'num_embeddings_per_dim': num_embeddings_per_dim
                    }
                    results.append((patches[i], attn_masks_flat[i], metadata))

            elif self.processing_function == 'resized':
                # Resized processing doesn't fit the "patching" paradigm well for DataLoader.
                # It processes the *entire* resized image at once.
                # You *could* treat the single resized image as *one* "patch".
                target_size = tuple(self.config['input_shape'])
                img_data_bc = np.expand_dims(img_data, axis=(0, 1))
                normalized_np = apply_normalization(scheme=NormalizationScheme.Z_SCORE, image=img_data_bc, target_dtype=np.float32)
                img_tensor = torch.from_numpy(normalized_np) # CPU

                if img_tensor.shape[-3:] != target_size:
                    img_tensor_resized = F.interpolate(img_tensor, size=target_size, mode='trilinear', align_corners=False)
                else:
                    img_tensor_resized = img_tensor

                # Create a dummy attention mask (all ones) if model requires it
                num_tokens = np.prod([t // v for t, v in zip(target_size, self.vit_patch_size)])
                attn_mask_flat = torch.ones(num_tokens, dtype=torch.bool)

                metadata = {
                        'image_id': image_id,
                        'region_key': 'resized',
                        'patch_index': 0, # Only one "patch"
                        'total_patches': 1,
                        'original_shape': original_full_shape, # Store original shape if needed later
                        'affine': affine,
                        'num_patches_dim': [1, 1, 1],
                        # TODO: derive this from the input_size / vit_embed_patch_size
                        'num_embeddings_per_dim': [24, 24, 24]
                }
                # Squeeze batch dim if added
                results.append((img_tensor_resized.squeeze(0), attn_mask_flat, metadata))


            elif self.processing_function == 'region':
                seg_path = get_seg_path_from_img_path(img_path, self.raw2rocket, self.segmentation_dir)
                if not os.path.exists(seg_path):
                    print(f"Warning: Seg not found for {img_path}, skipping.")
                    return [] # Return empty list if seg missing

                seg_data, _ = nrrd.read(seg_path)

                processed_groups = set()
                unique_mask_indices = np.unique(seg_data)
                if 0 in unique_mask_indices:
                    unique_mask_indices = unique_mask_indices[unique_mask_indices != 0]

                for mask_idx in unique_mask_indices:
                    mask_idx = int(mask_idx)
                    result_mask_check = _mask_idx_in_dict_of_interesting_masks(mask_idx, self.set_of_interesting_masks) # Use standalone helper

                    if result_mask_check:
                        class_name, current_set_of_indices = result_mask_check
                        if class_name in processed_groups:
                            continue

                        combined_mask = np.isin(seg_data, current_set_of_indices)
                        if not np.any(combined_mask):
                            processed_groups.add(class_name)
                            continue

                        coords = np.argwhere(combined_mask)
                        x_min, y_min, z_min = coords.min(axis=0)
                        x_max, y_max, z_max = coords.max(axis=0) + 1
                        img_data_cropped = img_data[x_min:x_max, y_min:y_max, z_min:z_max]
                        combined_mask_cropped = combined_mask[x_min:x_max, y_min:y_max, z_min:z_max]
                        cropped_shape = img_data_cropped.shape

                        # Patch the *cropped* region
                        patches, attn_masks_flat, _, _, num_patches, num_embeddings_per_dim = self._preprocess_and_patch(
                            img_data_cropped, affine, cropped_shape, foreground_mask=combined_mask_cropped
                        )

                        total_patches = patches.shape[0]
                        for i in range(total_patches):
                            metadata = {
                                'image_id': image_id,
                                'region_key': class_name,
                                'patch_index': i,
                                'total_patches': total_patches,
                                'original_shape': cropped_shape, # Shape of the *cropped region*
                                'affine': affine, # Affine relates to the *original* image space
                                'num_patches_dim': num_patches,
                                'num_embeddings_per_dim': num_embeddings_per_dim
                            }
                            results.append((patches[i], attn_masks_flat[i], metadata))

                        processed_groups.add(class_name)

            # Return a list of all patches generated from this image/its regions
            return results

        except Exception as e:
            print(f"ERROR processing {img_path} in Dataset worker: {e}")
            import traceback
            traceback.print_exc()
            return [] # Return empty list on error

# Helper from your original code needed by Dataset
def _mask_idx_in_dict_of_interesting_masks(mask_idx: int, interesting_masks: Dict[str, Union[int, List[int]]]) -> Union[bool, Tuple[str, List[int]]]:
    """Checks if a mask index belongs to any group in the interesting_masks dict."""
    if interesting_masks is None: return False
    for key, value in interesting_masks.items():
        indices = value if isinstance(value, list) else [value]
        if mask_idx in indices:
            return key, indices # Return group name and all indices in that group
    return False


def collate_patches(batch):
    """
    Collates list of lists of (patch, mask, metadata) tuples.
    'batch' is a list where each element is the output of ImagePatchDataset.__getitem__
    (i.e., a list of (patch, mask, metadata) for one image/its regions).
    """
    patches = []
    attn_masks = []
    metadata_list = []

    for item_list in batch: # item_list corresponds to one image/its regions
        for patch, mask, meta in item_list: # Iterate through patches of that image/region
            patches.append(patch)
            attn_masks.append(mask)
            metadata_list.append(meta)

    if not patches: # Handle case where batch is empty or all items failed
        # Return empty tensors of expected shape structure, but size 0 in batch dim
        # Or handle this upstream in the loop
        return None, None, []


    # Stack tensors: Requires patches and masks to have consistent shapes
    # Patch: (N, C, pH, pW, pD) -> (B_patch, C, pH, pW, pD)
    # Mask: (N, num_tokens) -> (B_patch, num_tokens)
    patches_batch = torch.stack(patches, dim=0)
    attn_masks_batch = torch.stack(attn_masks, dim=0)

    return patches_batch, attn_masks_batch, metadata_list

# --- Helper Functions ---

def load_yaml(file_path: str):
    """Loads a YAML file."""
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found at {file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {str(e)}")

def _create_name_mapping_rocket_raw(rocket_log_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Creates mapping between ROCKET IDs and raw filenames."""
    print(f"Loading ROCKET mapping from: {rocket_log_path}")
    if not os.path.exists(rocket_log_path):
        raise FileNotFoundError(f"ROCKET log file not found: {rocket_log_path}. Ensure --rocket_log_path is correct.")
    rocket_data = load_yaml(rocket_log_path)

    # Create mapping: ROCKET ID -> Raw Filename Stem (without _0000.nii.gz)
    rocket2raw = {}
    for d in rocket_data:
        if 'image_id' in d and 'image_filepaths' in d and d['image_filepaths']:
            # Handle potential .nii.gz and _0000 suffix
            raw_stem = Path(d['image_filepaths'][0]).name
            if raw_stem.endswith(".nii.gz"):
                raw_stem = raw_stem[:-7]
            if raw_stem.endswith("_0000"):
                raw_stem = raw_stem[:-5]
            rocket2raw[d['image_id']] = raw_stem
        else:
            warnings.warn(f"Skipping entry in rocket log due to missing keys: {d}")


    # Create reverse mapping: Raw Filename Stem -> ROCKET ID
    raw2rocket = {v: k for k, v in rocket2raw.items()}

    if not rocket2raw:
        warnings.warn(f"ROCKET mapping created from {rocket_log_path} is empty!")

    return rocket2raw, raw2rocket

def get_seg_path_from_img_path(img_path: str, raw2rocket: dict, segmentation_base_dir: str) -> str:
    """Constructs the segmentation file path based on the image path and mapping."""
    img_name = Path(img_path).name
    raw_stem = img_name
    if raw_stem.endswith(".nii.gz"):
        raw_stem = raw_stem[:-7]
    else:
        # img_path already adheres to rocket naming
        segmentation_path = Path(segmentation_base_dir) / raw_stem
        if segmentation_path.exists():
            return str(segmentation_path)
        raw_stem = Path(raw_stem).stem

    if raw_stem.endswith("_0000"):
        raw_stem = raw_stem[:-5]

    if raw_stem not in raw2rocket:
        raise KeyError(f"Raw image stem '{raw_stem}' (derived from {img_path}) not found in raw2rocket mapping. Check ROCKET log and input files.")

    rocket_name = raw2rocket[raw_stem]
    # Assume segmentation files are directly under segmentation_base_dir named as {rocket_name}.nrrd
    # Adjust this logic if your segmentation files are organized differently (e.g., in subdirs)
    segmentation_path = os.path.join(segmentation_base_dir, f"{rocket_name}.nrrd")
    return segmentation_path

# --- Model and Patching ---

# This should be loadable from primus.py
# def setup_model(base_path: str) -> Tuple[PretrainedEvaEncFromEvaMAE, dict]:
def setup_model(base_path: str):
    import primus
    model, config = primus.setup_model(base_path)
    return model, config
    """Loads the model and configuration from the specified base path."""
    plans_file = os.path.join(base_path, "plans.json")
    checkpoint_path = os.path.join(base_path, "fold_all", "checkpoint_final.pth")

    if not os.path.exists(plans_file):
        raise FileNotFoundError(f"Plans file not found: {plans_file}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading model config from: {plans_file}")
    print(f"Loading model weights from: {checkpoint_path}")

    config = get_config_from_plans(plans_file)
    model = PretrainedEvaEncFromEvaMAE(
        input_channels=1,
        output_channels=1, # This might be unused if only encoder is needed
        pretrained_weights_file=checkpoint_path,
        **config
    )
    model.eval()
    return model, config


def split_image_into_patches(img: torch.Tensor, patch_size: List[int], vit_patch_size: List[int], foreground_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Splits a 3D image tensor into patches for ViT processing, handling padding and optional foreground masking.

    Args:
        img: Input image tensor (B, C, H, W, D). Currently expects B=1.
        patch_size: The target size of the large patches (e.g., [192, 192, 192]).
        vit_patch_size: The size of the ViT's internal patches (e.g., [8, 8, 8]).
        foreground_mask: Optional foreground mask (H, W, D) used to ensure foreground patches are processed.

    Returns:
        Tuple containing:
        - patches: Tensor of image patches (N, B, C, pH, pW, pD).
        - attn_mask: Attention mask indicating valid ViT tokens within each patch (N, aH, aW, aD).
        - num_patches: List containing the number of patches along each dimension [nH, nW, nD].
    """
    assert img.ndim == 5, f"Input image must be 5D (B, C, H, W, D), got {img.ndim}"
    assert img.shape[0] == 1, f"Currently only supports batch size 1 for splitting, got {img.shape[0]}" # FIXME if batch > 1 needed

    img_shape = img.shape[-3:] # (H, W, D)
    device = img.device

    # Calculate number of patches in each dimension
    num_patches = [int(np.ceil(s / ps)) for s, ps in zip(img_shape, patch_size)]

    # Calculate total ViT tokens per dimension after padding
    total_vit_tokens_dim = [nps * int(ps / vps) for nps, ps, vps in zip(num_patches, patch_size, vit_patch_size)]

    # Initialize attention mask for ViT tokens (assumes all tokens are initially valid)
    attn_mask = torch.ones(total_vit_tokens_dim, dtype=torch.bool, device=device)

    # Calculate padding needed for the *image* to be divisible by patch_size
    padding = [(ps - s % ps) % ps for s, ps in zip(img_shape, patch_size)] # Correct padding calculation

    # Calculate how many *ViT tokens* correspond to the padding and mask them out
    tokens_to_mask = [int(np.floor(pad / vps)) for pad, vps in zip(padding, vit_patch_size)]

    if tokens_to_mask[0] > 0:
        attn_mask[-tokens_to_mask[0]:, :, :] = False
    if tokens_to_mask[1] > 0:
        attn_mask[:, -tokens_to_mask[1]:, :] = False
    if tokens_to_mask[2] > 0:
        attn_mask[:, :, -tokens_to_mask[2]:] = False

    # Pad image tensor: F.pad expects padding in order (last_dim_begin, last_dim_end, second_last_dim_begin, ...)
    pad_dims = [p for pair in zip([0]*3, padding[::-1]) for p in pair] # (pad_D_begin, pad_D_end, pad_W_begin, ...) -> (0, pad_z, 0, pad_y, 0, pad_x)
    img_padded = F.pad(img, pad_dims, mode='constant', value=0) # Pad with 0

    # Rearrange image into patches of size patch_size
    # B C (nps_x x) (nps_y y) (nps_z z) -> B C (nps_x nps_y nps_z) x y z  (where nps_i are num_patches; x,y,z are patch_size)
    patches = rearrange(img_padded, 'B C (nps_x x) (nps_y y) (nps_z z) -> B C (nps_x nps_y nps_z) x y z',
                        nps_x=num_patches[0], nps_y=num_patches[1], nps_z=num_patches[2],
                        x=patch_size[0], y=patch_size[1], z=patch_size[2])

    # Rearrange the attention mask to match the patch structure
    # (m ax) (n ay) (o az) -> (m n o) ax ay az (where ax = patch_size[0]/vit_patch_size[0], etc.)
    reduced_ps = [int(p / vps) for p, vps in zip(patch_size, vit_patch_size)]
    attn_mask = rearrange(attn_mask, '(m ax) (n ay) (o az) -> (m n o) ax ay az',
                          m=num_patches[0], n=num_patches[1], o=num_patches[2],
                          ax=reduced_ps[0], ay=reduced_ps[1], az=reduced_ps[2])


    if foreground_mask is not None:
        # Pad foreground mask to match padded image dimensions
        # Assumes foreground_mask is (H, W, D)
        assert foreground_mask.shape == img_shape, f"Foreground mask shape {foreground_mask.shape} does not match image shape {img_shape}"
        fg_mask_padded = F.pad(foreground_mask.unsqueeze(0).unsqueeze(0), # Add B, C dims
                               pad_dims, mode='constant', value=0).squeeze(0).squeeze(0) # Remove B,C dims

        # Create a (patch_size[i] / vit_patch_size[i])^3 grid of patches, each of size vit_patch_size
        # You can think of this grid as a grid of ViT tokens within the larger patch
        # e.g. if patch_size = (192, 192, 192) and vit_patch_size = (8, 8, 8), then
        # each token will be of size (8, 8, 8) and there will be (192/8) = 24 tokens in each dimension
        # 24 x 24 x 24 = 13824 tokens in total
        fg_mask_vit_grid = rearrange(fg_mask_padded, '(mt ax) (nt ay) (ot az) -> (mt nt ot) ax ay az',
                                     ax=vit_patch_size[0], # Total tokens H = mt * ax
                                     ay=vit_patch_size[1], # Total tokens W = nt * ay
                                     az=vit_patch_size[2]) # Total tokens D = ot * az

        # Check if *any* voxel within each ViT token region is foreground (part of the mask)
        # Shape: (Total_ViT_Tokens_H * Total_ViT_Tokens_W * Total_ViT_Tokens_D)
        # e.g. (24 * 24 * 24, 8, 8, 8) -> (24 * 24 * 24)
        fg_token_active = torch.any(fg_mask_vit_grid != 0, dim=(-3, -2, -1)) # Check if any voxel in the vx*vy*vz region is > 0

        # Reshape this activation map back to the grid dimensions (Total_ViT_Tokens_H, Total_ViT_Tokens_W, Total_ViT_Tokens_D)
        fg_token_activation_map = rearrange(fg_token_active, '(mt nt ot) -> mt nt ot',
                                            mt=total_vit_tokens_dim[0], # total H vit tokens
                                            nt=total_vit_tokens_dim[1], # total W vit tokens
                                            ot=total_vit_tokens_dim[2]) # total D vit tokens

        # Rearrange this map to match the attention mask structure (N, ax, ay, az)
        fg_mask_for_attn = rearrange(fg_token_activation_map, '(m ax) (n ay) (o az) -> (m n o) ax ay az',
                                     m=num_patches[0], n=num_patches[1], o=num_patches[2])

        # Update the attention mask: A token is attended if it's not padding OR if it corresponds to a foreground region
        attn_mask = attn_mask | fg_mask_for_attn.to(attn_mask.device) # Use logical OR

    # Permute patches: (B, C, N, pH, pW, pD) -> (N, B, C, pH, pW, pD)
    # This makes it easier to iterate through patches
    patches = patches.permute(2, 0, 1, 3, 4, 5)

    return patches, attn_mask, num_patches


def combine_patches_into_image(embeddings: List[torch.Tensor], attn_masks: List[torch.Tensor], num_patches: List[int], original_shape_no_batch_channel: Tuple[int, int, int]):
    """Combines patch embeddings back into a single image, removing padding."""
    if not embeddings:
        return torch.empty((1, 0) + original_shape_no_batch_channel, device=attn_masks.device) # Return empty tensor matching expected dims

    # Stack embeddings: List of (B, C, H, W, D) -> (N, B, C, H, W, D)
    embeddings_stacked = torch.stack(embeddings)
    attn_masks_stacked = torch.stack(attn_masks)
    # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Stacked embeddings shape: {embeddings_stacked.shape}")
    N, C, H, W, D = embeddings_stacked.shape # Get dimensions from embeddings

    # Rearrange embeddings back into the padded grid structure
    # (N) B C H W D -> B C (X H) (Y W) (Z D)  where N = X*Y*Z
    # print(f"num_patches: {num_patches}")
    whole_image_with_padding = rearrange(embeddings_stacked, '(X Y Z) C H W D -> C (X H) (Y W) (Z D)',
                                        X=num_patches[0], Y=num_patches[1], Z=num_patches[2])

    attn_mask_image_reshaped = rearrange(attn_masks_stacked, '(X Y Z) ax ay az -> (X ax) (Y ay) (Z az)',
                                        X=num_patches[0], Y=num_patches[1], Z=num_patches[2])
    keep_mask = torch.nonzero(attn_mask_image_reshaped)
    min_indices = keep_mask.min(dim=0).values
    max_indices = keep_mask.max(dim=0).values
    whole_image_cropped_to_attention = whole_image_with_padding[:, min_indices[0]:max_indices[0]+1, min_indices[1]:max_indices[1]+1, min_indices[2]:max_indices[2]+1]
    return whole_image_cropped_to_attention


    # Calculate padding that was added (similar logic to split_image_into_patches)
    padded_shape = whole_image_with_padding.shape[-3:]
    padding = [(padded - orig) for padded, orig in zip(padded_shape, original_shape_no_batch_channel)]

    # Calculate slices to remove padding
    # We added padding only at the end of dimensions
    slices = [slice(None)] * 2 + [slice(0, orig) for orig in original_shape_no_batch_channel]

    # Crop the image to remove padding
    whole_image_cropped = whole_image_with_padding[tuple(slices)]

    # Sanity check final shape
    expected_shape = (B, C) + original_shape_no_batch_channel
    if whole_image_cropped.shape != expected_shape:
        warnings.warn(f"Warning: Combined image shape {whole_image_cropped.shape} does not match expected original shape {expected_shape}. Cropping might be incorrect.")

    return whole_image_cropped


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
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def process_image_resized(
        img_path: str,
        # model: Union[PretrainedEvaEncFromEvaMAE, DDP],
        model,
        config: dict,
        device: torch.device,
        vit_patch_size: List[int] # Added for consistency, though not directly used here
    ) -> Tuple[torch.Tensor, np.ndarray]:
    """Processes the image by resizing it to the model's expected input patch size."""
    target_size = tuple(config['input_shape'])
    print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Processing {Path(img_path).name} by resizing to {target_size}.")
    try:
        img_data, affine = load_image_data_and_affine(img_path) # Load image data and affine
        original_shape = img_data.shape # H, W, D

        # Add batch and channel dimensions
        img_data_bc = np.expand_dims(img_data, axis=(0, 1))

        # Apply normalization
        normalized_np = apply_normalization(
            scheme=NormalizationScheme.Z_SCORE, # Or use scheme from config if available
            image=img_data_bc,
            target_dtype=np.float32,
            use_mask_for_norm=False,
        )

        # Convert to tensor
        img_tensor = torch.from_numpy(normalized_np).to(device)
        # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Tensor shape before resize: {img_tensor.shape}")

        # Resize image to match target size
        if img_tensor.shape[-3:] != target_size:
            img_tensor_resized = F.interpolate(
                img_tensor,
                size=target_size,
                mode='trilinear', # Use trilinear for potentially better results than default linear
                align_corners=False # Generally recommended for medical images
            )
            # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Resized tensor shape: {img_tensor_resized.shape}")
        else:
            img_tensor_resized = img_tensor
            # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Image already at target size, skipping resize.")


        # Generate embeddings
        with torch.no_grad():
            # For resized input, the model expects the whole (resized) image.
            # Does the model accept an attention mask here? Assuming not for the simple resize case.
            # If it DOES require a mask (e.g., always), we need to create a mask of all ones.
            # Check model's forward signature. Assuming model(x) is valid for resized input.
            embeddings = model(img_tensor_resized)
            # If model needs a mask:
            # num_tokens = np.prod([t // v for t, v in zip(target_size, vit_patch_size)])
            # mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            # embeddings = model(img_tensor_resized, mask)

        print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Generated resized embedding shape: {embeddings.shape}")
        return embeddings.cpu(), affine

    except Exception as e:
        print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Error processing resized {img_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty tensor on error
        embedding_dim = 864 # Placeholder
        try:
            affine = nib.load(img_path).affine
            return torch.empty((1, embedding_dim) + target_size, dtype=torch.float32), affine
        except Exception:
            return torch.empty((1, embedding_dim, 0, 0, 0), dtype=torch.float32), np.eye(4)


# --- DDP Setup ---

def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
    print(f"Rank {rank}: Initializing DDP with world size {world_size} master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # Use NCCL backend for Nvidia GPUs
    torch.cuda.set_device(rank)
    print(f"Rank {rank}: DDP Initialized. Using GPU {torch.cuda.current_device()}")


def cleanup_ddp():
    """Destroys the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP process group destroyed.")


# --- Main Execution Logic ---

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate image embeddings using a pre-trained model, with optional DDP support.")
    parser.add_argument("--base_path", required=True, type=str, help="Path to the model directory containing plans.json and fold_all/checkpoint_final.pth")
    parser.add_argument("--output_base", required=True, type=str, help="Base directory to save the output embeddings (.npz files).")
    parser.add_argument("--input_dirs", required=True, nargs='+', type=str, help="List of input directories containing .nii.gz files.")
    parser.add_argument("--processing_function", required=True, type=str, choices=['native', 'resized', 'region'], help="Embedding generation method: 'native' (patch-based full image), 'resized' (resize full image), 'region' (patch-based region extraction).")
    parser.add_argument("--vit_patch_size", type=int, nargs=3, default=[8, 8, 8], help="ViT internal patch size (e.g., 8 8 8).")

    # Arguments specific to 'region' processing
    parser.add_argument("--segmentation_dir", type=str, default=None, help="Directory containing segmentation files (.nrrd), required if processing_function='region'. Assumes filenames match ROCKET IDs.")
    parser.add_argument("--dataset_json_path", type=str, default=None, help="Path to dataset.json containing label mappings, required if processing_function='region'.")
    parser.add_argument("--rocket_log_path", type=str, default=None, help="Path to ROCKET successful_conversions.yml log file, required if processing_function='region'.")

    # DDP related (optional, defaults to single process if not launched with torchrun)
    # These are typically set by the launcher, but can be specified for debugging
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', -1), help='Local rank for distributed training')
    parser.add_argument("--dataloader_batch_size", type=int, default=1, help="Number of *images* per batch for DataLoader to process.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for DataLoader.")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch per worker.")

    parser.add_argument("--max_gpu_patch_batch_size", type=int, default=1, help="Batch size processed on the GPU.")


    args = parser.parse_args()

    # Validation for region processing
    if args.processing_function == 'region':
        if not args.segmentation_dir:
            parser.error("--segmentation_dir is required when processing_function='region'")
        # if not args.dataset_json_path:
        #      parser.error("--dataset_json_path is required when processing_function='region'")
        if not args.rocket_log_path:
            parser.error("--rocket_log_path is required when processing_function='region'")

    return args

def main(args):
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['LOCAL_RANK']) # Get rank from environment variable
        setup_ddp(rank, world_size)
        is_ddp = True
        device = torch.device(f'cuda:{rank}')
    else:
        # Single process mode
        rank = 0
        world_size = 1
        is_ddp = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Running in single process mode.")


    try:
        # --- Setup Model --- (Keep as is)
        print(f"Rank {rank}: Loading model...")
        model, config = setup_model(args.base_path)
        model = model.to(device)
        if is_ddp:
            # Adjust find_unused_parameters based on whether 'resized' mode is used
            # If resized is used, the main encoder path might be the only one used.
            # If native/region, the patching/embedding logic is used. Check model internals.
            find_unused = True # Safer default, check if performance hit exists
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused)
        model.eval()

        # --- Collect Files --- (Only need the list of files)
        all_img_files_paths = []
        output_dir_map = {} # Map input dir to output dir
        for input_dir in args.input_dirs:
            dir_name = os.path.basename(input_dir.rstrip('/'))
            output_dir = os.path.join(args.output_base, dir_name)
            if rank == 0:
                os.makedirs(output_dir, exist_ok=True)
            output_dir_map[input_dir] = output_dir # Store for later lookup

            if os.path.isdir(input_dir):
                for f in os.listdir(input_dir):
                    if f.endswith(('.nii.gz', '.nrrd')): # Tuple for endswith
                        img_path = os.path.join(input_dir, f)
                        all_img_files_paths.append(img_path) # Just the path
            else:
                print(f"Warning: Input directory not found: {input_dir}")

        if is_ddp:
            dist.barrier() # Ensure rank 0 created dirs


        # --- Create Dataset and DataLoader ---
        # TODO: get vit patch size from the model config
        dataset_kwargs = {
            'config': config,
            'vit_patch_size': config['architecture_plans']['arch_kwargs']["patch_embed_size"],
            'processing_function': args.processing_function
        }
        if args.processing_function == 'region':
            dataset_kwargs.update({
                'segmentation_dir': args.segmentation_dir,
                'dataset_json_path': args.dataset_json_path, # Pass path or loaded data
                'rocket_log_path': args.rocket_log_path, # Pass path or loaded data
                # 'set_of_interesting_masks': None # Pass custom dict if needed
            })
        elif args.processing_function == 'resized':
            # Ensure vit_patch_size is passed if needed for dummy mask
            pass


        dataset = ImagePatchDataset(all_img_files_paths, **dataset_kwargs)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

        # batch_size here means how many *images* are processed by workers concurrently
        # before their patches are collated. Adjust based on memory/CPU cores.
        # Increase num_workers to utilize more CPU cores for pre-processing.
        # pin_memory=True can speed up CPU->GPU transfer if workers load to pinned memory.
        dataloader = DataLoader(
            dataset,
            batch_size=args.dataloader_batch_size, # Add a new arg for this! e.g., 4 or 8
            shuffle=False, # Shuffle is handled by DistributedSampler
            sampler=sampler,
            num_workers=args.num_workers, # Add a new arg! e.g., 4, 8, 16
            collate_fn=collate_patches,
            pin_memory=True, # Generally good if using GPUs
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else 2 # Add new arg, e.g., 2 or 4
        )

        # --- Process Batches ---
        # Structure to hold results temporarily before saving
        # Key: (image_id, region_key)
        # Value: {'embeddings': [None]*total_patches, 'affine': affine, 'original_shape': shape, 'total_patches': N, 'output_path_base': path}
        results_buffer = defaultdict(lambda: {'embeddings': [], 'affine': None, 'original_shape': None, 'total_patches': 0, 'output_path_base': None})

        iterable = dataloader
        if rank == 0:
            # Total batches is tricky since it depends on patches, maybe use total images / dataloader_batch_size
            total_batches = len(dataloader)
            iterable = tqdm(dataloader, total=total_batches, desc=f"Processing Batches (Rank 0/{world_size})")

        processed_patch_count = 0
        for batch_data in iterable:
            if batch_data is None: # Skip if collate_fn returned None (empty batch)
                continue

            patches_batch, attn_masks_batch, metadata_list = batch_data

            if patches_batch is None or patches_batch.numel() == 0:
                if rank == 0: print("Skipping empty batch")
                continue

            # --- Model Inference ---
            patches_batch = patches_batch.to(device, non_blocking=True) # non_blocking with pin_memory=True
            attn_masks_batch = attn_masks_batch.to(device, non_blocking=True)

            current_collated_batch_size = patches_batch.shape[0]

            # --- Model Inference with Mini-Batching ---
            all_embeddings_list = [] # To store results from mini-batches
            all_attention_masks = []
            max_gpu_batch_size = args.max_gpu_patch_batch_size
            with torch.no_grad():
                for i in range(0, current_collated_batch_size, max_gpu_batch_size):
                    start_idx = i
                    end_idx = min(i + max_gpu_batch_size, current_collated_batch_size)

                    # Get the current mini-batch
                    mini_batch_patches = patches_batch[start_idx:end_idx]
                    mini_batch_attn_masks = attn_masks_batch[start_idx:end_idx]

                    # Move *only the mini-batch* to GPU
                    mini_batch_patches = mini_batch_patches.to(device, non_blocking=True)
                    mini_batch_attn_masks = mini_batch_attn_masks.to(device, non_blocking=True)

                    # Add dimensions for num_heads and target_seq_len to conform to scaled_dot_product impl
                    # (B, H, W, D) -> (B, 1, 1, Seq_len)
                    mini_batch_attn_masks_correct_shape = rearrange(mini_batch_attn_masks, 'B H W D -> B 1 1 (H W D)')

                    # Run model on the mini-batch
                    # Ensure model's forward takes flattened mask if that's what you designed
                    mini_batch_embeddings = model(mini_batch_patches, mini_batch_attn_masks_correct_shape)

                    # Append result to list (move to CPU to free GPU VRAM)
                    all_embeddings_list.extend(mini_batch_embeddings.cpu())
                    all_attention_masks.extend(mini_batch_attn_masks.cpu())

                    # Optional: Clear GPU cache if memory pressure is extreme, but usually not needed
                    # torch.cuda.empty_cache()

            # Concatenate results from all mini-batches for this collated batch
            if all_embeddings_list:
                batch_embeddings_cpu = all_embeddings_list
            else:
                # Handle case where the loop didn't run (e.g., initial batch size was 0)
                raise RuntimeError("No embeddings generated for this batch.")

            # --- Collect Results ---
            processed_patch_count += len(batch_embeddings_cpu)

            # Iterate through the results and metadata from this batch
            for i, meta in enumerate(metadata_list):
                image_id = meta['image_id']
                region_key = meta['region_key']
                patch_idx = meta['patch_index']
                total_patches = meta['total_patches']
                affine = meta['affine']
                original_shape = meta['original_shape'] # Shape used for patching (full or cropped)
                num_patches_dim = meta['num_patches_dim']

                result_key = (image_id, region_key)

                # Initialize entry if first time seeing this image/region
                if result_key not in results_buffer:
                    results_buffer[result_key]['embeddings'] = [None] * total_patches
                    results_buffer[result_key]['attn_masks'] = [None] * total_patches
                    results_buffer[result_key]['affine'] = affine
                    results_buffer[result_key]['original_shape'] = original_shape
                    results_buffer[result_key]['total_patches'] = total_patches
                    results_buffer[result_key]['num_patches_dim'] = num_patches_dim

                    # Determine output path base only once
                    # Find original input dir from image_id/path (might need mapping back)
                    # This part is tricky, maybe pass img_path in metadata?
                    # Let's assume img_path was in metadata for simplicity here:
                    # img_path_orig = meta['img_path'] # <-- Add this to dataset's metadata
                    # input_dir = os.path.dirname(img_path_orig)
                    # base_name = Path(img_path_orig).stem.replace('_0000','')
                    # output_dir = output_dir_map[input_dir]
                    # results_buffer[result_key]['output_path_base'] = os.path.join(output_dir, base_name)
                    # FIXME: Need a reliable way to get output_path_base

                    # --- Simplified Output Path (assuming image_id is filename stem) ---
                    # Find the input dir the image_id belongs to (requires searching input_dirs)
                    found_dir = None
                    for indir in args.input_dirs:
                        # Check if the original file exists (e.g., image_id + ".nii.gz")
                        # This is inefficient, better pass img_path in metadata
                        potential_path_nii = os.path.join(indir, image_id + ".nii.gz")
                        potential_path_nrrd = os.path.join(indir, image_id + ".nrrd")
                        if os.path.exists(potential_path_nii) or os.path.exists(potential_path_nrrd):
                            found_dir = indir
                            break
                    if found_dir:
                        output_dir = output_dir_map[found_dir]
                        results_buffer[result_key]['output_path_base'] = os.path.join(output_dir, image_id)
                    else:
                        print(f"Rank {rank} Error: Could not determine output path for {image_id}")
                        # Skip this result or assign a default path
                        results_buffer[result_key]['output_path_base'] = os.path.join(args.output_base, "unknown", image_id)
                        os.makedirs(os.path.dirname(results_buffer[result_key]['output_path_base']), exist_ok=True)
                    # --- End Simplified Output Path ---


                # Store the embedding (ensure index is within bounds)
                if patch_idx < len(results_buffer[result_key]['embeddings']):
                    results_buffer[result_key]['embeddings'][patch_idx] = batch_embeddings_cpu[i]
                    results_buffer[result_key]['attn_masks'][patch_idx] = all_attention_masks[i]
                else:
                    print(f"Rank {rank} Error: Patch index {patch_idx} out of bounds for {result_key} (total: {total_patches})")


                # --- Check for Completion and Save ---
                # Check if all embeddings for this key have arrived
                current_embeddings = results_buffer[result_key]['embeddings']
                if all(e is not None for e in current_embeddings):
                    # print(f"Rank {rank}: Image/Region {result_key} complete ({total_patches} patches). Combining...")
                    # Combine: Need combine_patches_into_image
                    # Pass the list of embedding tensors, dummy attn_masks (not needed for combine),
                    # num_patches list, and original_shape
                    combined_embedding = combine_patches_into_image(
                        current_embeddings, # List of tensors [(EmbedDim, eH, eW, eD), ...]
                        attn_masks=results_buffer[result_key]['attn_masks'], # Not used by combine func based on its signature
                        # Need num_patches (dims) from split_image_into_patches output if combine needs it
                        # Let's assume combine_patches_into_image was adapted or doesn't need num_patches_dims
                        num_patches = results_buffer[result_key]['num_patches_dim'], # Placeholder - check if combine_patches needs this
                        original_shape_no_batch_channel=results_buffer[result_key]['original_shape']
                    )

                    # Save the combined embedding
                    output_path_base = results_buffer[result_key]['output_path_base']
                    affine_to_save = results_buffer[result_key]['affine']

                    if combined_embedding.numel() > 0:
                        # Determine final output path based on region_key
                        if region_key == 'full' or region_key == 'resized':
                            output_path = output_path_base + "_embedding.npz"
                        else: # Region processing
                            output_path = f"{output_path_base}_{region_key}_embedding.npz"

                        print(f"Rank {rank}: Saving combined embedding to {output_path}")
                        np.savez_compressed(
                            output_path,
                            embeddings=combined_embedding.numpy(), # Should be (B=1, C, H, W, D)
                            affine=affine_to_save,
                            # original_file= # Need original filename, pass via metadata
                            region_key=region_key if region_key not in ['full', 'resized'] else None
                        )
                    else:
                        print(f"Rank {rank}: Skipping save for {result_key} - combined embedding is empty.")


                    # Remove completed entry from buffer
                    del results_buffer[result_key]


            # Optional: Add a check/log for buffer size to prevent memory issues if saving is slow
            if rank == 0 and len(results_buffer) > 50: # Example threshold
                print(f"Warning: Results buffer size = {len(results_buffer)}")


        # --- Final Check ---
        # Ensure all data processed (especially important without DDP or if dataset size isn't perfectly divisible)
        if is_ddp:
            dist.barrier() # Wait for all ranks to finish processing batches

        # Gather remaining buffer items across ranks (complex with defaultdict)
        # Simpler: Assume DDP handles distribution fairly and buffer should be mostly empty
        # Or implement manual gathering if needed
        if rank == 0 and results_buffer:
            print(f"Warning: {len(results_buffer)} items remaining in results buffer after loop. Might indicate incomplete images.")
            # Optionally try processing/saving them here if possible

        if rank == 0: print(f"Processed approx {processed_patch_count} patches total across all ranks.")


    finally:
        # --- DDP Cleanup ---
        if is_ddp:
            cleanup_ddp()


if __name__ == "__main__":
    args = parse_arguments()
    # try:
    main(args)
    print("Processing finished.")
    # except KeyboardInterrupt:
    #     print("\nProcess interrupted by user. Cleaning up...")
    #     cleanup_ddp() # Attempt cleanup on interrupt
    # except Exception as e:
    #     print(f"\nAn unexpected error occurred: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    #     cleanup_ddp() # Attempt cleanup on error
    #     exit(1) # Exit with error code