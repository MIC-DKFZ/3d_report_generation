from typing import List, Optional, Union, Dict, Tuple
import resenc
import nibabel as nib
import nrrd
import yaml
import json
import torch
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import blosc2
import traceback
from einops import rearrange
import random

from prep.normalization_schemes import (
    apply_normalization,
    NormalizationScheme,
)


import torch.distributed as dist
import os


def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl", init_method="env://")
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


def save_embeddings_blosc2(
    embeddings, output_path, clevel: int = 8, codec=blosc2.Codec.ZSTD
):
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

    bytes_saved = blosc2.save_array(
        embeddings_np, output_path, mode="w", cparams=cparams
    )
    return


def split_image_into_patches(
    img,
    patch_size: List[int],
    vit_patch_size: List[int],
    foreground_mask: torch.Tensor = None,
):
    # Get image shape
    img_shape = img.shape[-3:]  # Shape should be (B, C, H, W, D) -> (H, W, D)

    # Calculate number of patches in each dimension
    num_patches = [int(np.ceil(s / p)) for s, p in zip(img_shape, patch_size)]

    attn_mask = torch.ones(
        [
            nps * int(ps / vps)
            for nps, ps, vps in zip(num_patches, patch_size, vit_patch_size)
        ],
        dtype=torch.bool,
    )

    # Calculate padding for each dimension
    padding = [((p - s) % p) for s, p in zip(img_shape, patch_size)]
    tokens_to_mask = [
        int(np.floor(pad / vps)) for pad, vps in zip(padding, vit_patch_size)
    ]
    if tokens_to_mask[0] > 0:
        attn_mask[-tokens_to_mask[0] :, :, :] = False
    if tokens_to_mask[1] > 0:
        attn_mask[:, -tokens_to_mask[1] :, :] = False
    if tokens_to_mask[2] > 0:
        attn_mask[:, :, -tokens_to_mask[2] :] = False

    # Pad image to fit patches
    img_padded = F.pad(
        img,
        tuple([x for xs in [[0, p] for p in padding[::-1]] for x in xs]),
        mode="constant",
    )

    # Split image into patches
    patches = rearrange(
        img_padded,
        "B C (m x) (n y) (o z) -> B C (m n o) x y z",
        m=num_patches[0],
        n=num_patches[1],
        o=num_patches[2],
        x=patch_size[0],
        y=patch_size[1],
        z=patch_size[2],
    )

    # Split attention mask into patches
    reduced_ps = [int(p / vps) for p, vps in zip(patch_size, vit_patch_size)]
    attn_mask = rearrange(
        attn_mask,
        "(m x) (n y) (o z) -> (m n o) x y z",
        m=num_patches[0],
        n=num_patches[1],
        o=num_patches[2],
        x=reduced_ps[0],
        y=reduced_ps[1],
        z=reduced_ps[2],
    )

    if foreground_mask is not None:
        # Pad foreground mask to fit patches
        foreground_mask_padded = F.pad(
            foreground_mask,
            tuple([x for xs in [[0, p] for p in padding[::-1]] for x in xs]),
            mode="constant",
        )
        # This creates a 24x24x24 grid of patches, each size 8x8x8 (if vit_patch_size = 8 and patch_size / vit_patch_size = 24)
        foreground_mask_patches = rearrange(
            foreground_mask_padded,
            "(m x) (n y) (o z) -> (m n o) x y z",
            x=vit_patch_size[0],
            y=vit_patch_size[1],
            z=vit_patch_size[2],
        )
        # If any voxel in a 8x8x8 patch contains foreground, set the whole patch to foreground
        # (24 * 24 * 24, 8, 8, 8) -> (24 * 24 * 24)
        mask = torch.any(foreground_mask_patches != 0, dim=(-3, -2, -1))

        foreground_mask_for_mask = rearrange(
            mask,
            "(m n o) -> m n o",
            m=reduced_ps[0] * num_patches[0],
            n=reduced_ps[1] * num_patches[1],
            o=reduced_ps[2] * num_patches[2],
        )
        foreground_mask_match_attn_mask = rearrange(
            foreground_mask_for_mask,
            "(i m) (j n) (k o) -> (i j k) m n o",
            i=num_patches[0],
            j=num_patches[1],
            k=num_patches[2],
        ).to(attn_mask.device)
        attn_mask = attn_mask | foreground_mask_match_attn_mask.to(attn_mask.device)

    # (B, C, N, H, W, D) -> (N, B, C, H, W, D)
    return patches.permute(-4, 0, 1, -3, -2, -1), attn_mask, num_patches


def combine_patches_into_image(embeddings, attn_masks, num_patches):
    whole_image_with_padding = rearrange(
        torch.stack(embeddings),
        "(X Y Z) B C H W D -> B C (X H) (Y W) (Z D)",
        X=num_patches[0],
        Y=num_patches[1],
        Z=num_patches[2],
    )
    whole_attention_mask = rearrange(
        attn_masks,
        "(X Y Z) H W D -> (X H) (Y W) (Z D)",
        X=num_patches[0],
        Y=num_patches[1],
        Z=num_patches[2],
    )
    true_indices = torch.nonzero(whole_attention_mask)
    min_indices = true_indices.min(dim=0).values
    max_indices = true_indices.max(dim=0).values
    # + 1 since slice end is exclusive
    whole_image_cropped = whole_image_with_padding[
        :,
        :,
        min_indices[0] : max_indices[0] + 1,
        min_indices[1] : max_indices[1] + 1,
        min_indices[2] : max_indices[2] + 1,
    ]
    return whole_image_cropped


def load_yaml(file_path: str):
    try:
        with open(file_path, "r") as f:
            rocket_data = yaml.safe_load(f)
            return rocket_data
    except FileNotFoundError:
        raise f"Rocket data file not found at {file_path}"
    except yaml.YAMLError as e:
        raise f"Error parsing Rocket data file: {str(e)}"


def _create_name_mapping_rocket_raw(dataset_name: str = "CPE_data/AMOS-MM"):
    rocket_path = f"/home/r403k/ResourcesDKFZ/e230-thrp-data/mic_rocket/data/raw_converted/{dataset_name}/logs/successful_conversions.yml"
    rocket_data = load_yaml(rocket_path)

    # create mapping
    rocket2raw = {
        d["image_id"]: Path(d["image_filepaths"][0]).stem.split(".")[0]
        for d in rocket_data
    }
    raw2rocket = {v: k for k, v in rocket2raw.items()}
    return rocket2raw, raw2rocket


def process_image_center_crop(
    img_path, model, config, device="cuda" if torch.cuda.is_available() else "cpu"
):
    target_size = (192, 192, 192)

    blosc2_result = load_blosc2(img_path)
    img_data, _ = blosc2_result
    img = np.array(img_data)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    normalized_np = apply_normalization(
        scheme=NormalizationScheme.Z_SCORE,
        image=img,
        target_dtype=np.float32,
        use_mask_for_norm=False,
    )

    # Convert back to tensor
    img_tensor = torch.from_numpy(normalized_np).to(device)
    print(f"Tensor shape after adding batch/channel dims: {img_tensor.shape}")

    # img_tensor shape is [B, C, H, W, D]
    spatial_shape = img_tensor.shape[-3:]  # (H, W, D)

    # Calculate center crop coordinates
    center_coords = [s // 2 for s in spatial_shape]
    half_target = [t // 2 for t in target_size]

    # Calculate crop boundaries, ensuring we don't go outside image bounds
    crop_start = [max(0, c - h) for c, h in zip(center_coords, half_target)]
    crop_end = [
        min(s, c + h) for s, c, h in zip(spatial_shape, center_coords, half_target)
    ]

    # Adjust crop_start if crop_end - crop_start < target_size and we have room
    for i in range(3):
        actual_size = crop_end[i] - crop_start[i]
        if actual_size < target_size[i]:
            # Try to expand the crop if possible
            needed = target_size[i] - actual_size
            expand_start = min(needed // 2, crop_start[i])
            expand_end = min(needed - expand_start, spatial_shape[i] - crop_end[i])
            crop_start[i] -= expand_start
            crop_end[i] += expand_end

    # Extract center crop
    center_crop = img_tensor[
        :,
        :,
        crop_start[0] : crop_end[0],
        crop_start[1] : crop_end[1],
        crop_start[2] : crop_end[2],
    ]

    # we use split_image_into_patches to compute the attention mask (only one patch)
    patches, attn_masks, num_patches = split_image_into_patches(
        center_crop, [192, 192, 192], vit_patch_size=[8, 8, 8]
    )

    with torch.no_grad():
        batch_patches = patches.view(-1, *patches.shape[2:])
        batch_attn_masks = rearrange(attn_masks, "n h w d -> n 1 1 (h w d)").to(device)
        batch_embeddings = model(batch_patches, batch_attn_masks)
        embeddings = list(batch_embeddings.split(patches.shape[1], dim=0))
    embeddings = combine_patches_into_image(
        embeddings, attn_masks.to(device), num_patches
    )

    return embeddings.cpu(), None


@torch.inference_mode()
def process_image_native_res(
    img_path, model, config, device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Print debug information about patch size
    patch_size = tuple(config["recommended_downstream_patchsize"])
    try:
        blosc2_result = load_blosc2(img_path)
        img_data, _ = blosc2_result
        img = np.array(img_data)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        normalized_np = apply_normalization(
            scheme=NormalizationScheme.Z_SCORE,
            image=img,
            target_dtype=np.float32,
            use_mask_for_norm=False,
        )

        # Convert back to tensor
        img_tensor = torch.from_numpy(normalized_np).to(device)
        print(f"Tensor shape after adding batch/channel dims: {img_tensor.shape}")

        # Get patch size from config and ensure it's a tuple
        patch_size = tuple(
            config["recommended_downstream_patchsize"]
        )  # Should be (192, 192, 192)
        # FIXME: currenlty function only supports Batch size of 1
        patches, attn_masks, num_patches = split_image_into_patches(
            img_tensor, patch_size, vit_patch_size=[8, 8, 8]
        )

        # before: loop over windows
        # for i, (patch, attn_mask) in enumerate(zip(patches, attn_masks)):
        #     with torch.no_grad():
        #         #print(rearrange(attn_mask, "h w d -> 1 1 1 (h w d)").shape)
        #         print("patch", patch.shape)
        #         patch_embeddings = model(
        #             patch, rearrange(attn_mask, "h w d -> (h w d)").to(device)
        #         )
        #         embeddings.append(patch_embeddings)
        #         import napari, sys
        #         viewer = napari.Viewer()
        #         viewer.add_image(patch[0,0].detach().cpu().numpy(), name="patch")
        #         viewer.add_image(patch_embeddings[0,0].detach().cpu().numpy(), name="reconst")
        #         napari.run()
        #         sys.exit(0)

        # All sliding windows at once (output is the same except some small numerical error < 1e-5)
        with torch.no_grad():
            batch_patches = patches.view(-1, *patches.shape[2:])
            batch_attn_masks = rearrange(attn_masks, "n h w d -> n 1 1 (h w d)").to(
                device
            )
            batch_embeddings = model(batch_patches, batch_attn_masks)
            embeddings = list(batch_embeddings.split(patches.shape[1], dim=0))

        embeddings = combine_patches_into_image(
            embeddings, attn_masks.to(device), num_patches
        )

        return embeddings.cpu(), None

    except Exception as e:
        print(f"DEBUG: Exception occurred in process_image_native_res")
        print(f"DEBUG: Exception type: {type(e)}")
        print(f"DEBUG: Exception message: {str(e)}")
        print(f"Error in process_image for {img_path}: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


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
    space_directions = header["space directions"]
    space_origin = header["space origin"]

    # Handle cases where 'space directions' might have 'none' entries (e.g., for scalar data)
    valid_directions = []
    for direction in space_directions:
        if not np.isnan(
            direction
        ).any():  # or not any(d is None for d in direction) #removed this as it does raise a type error.
            valid_directions.append(direction)
    valid_directions = np.array(valid_directions)

    affine = np.eye(4)
    affine[: valid_directions.shape[0], : valid_directions.shape[1]] = valid_directions
    affine[:3, 3] = space_origin
    # The following line is needed as otherwise some files do have a shape which is not correct for multiplication
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
        _, ext_inner = os.path.splitext(
            img_path[:-3]
        )  # Remove .gz and get next extension
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
    dparams = {"nthreads": 1}

    mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}

    data_array = blosc2.open(urlpath=img_path, mode="r", dparams=dparams, **mmap_kwargs)
    print(f"blosc2 array loaded, dtype: {data_array.dtype}, shape: {data_array.shape}")

    return data_array, None


def process_image(
    img_path, model, config, device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Print debug information about patch size
    patch_size = tuple(config["recommended_downstream_patchsize"])
    # print(f"Target patch size from config: {patch_size}")
    try:
        img_data, affine = load_image_data_and_affine(
            img_path
        )  # Load image data and affine

        # Add batch and channel dimensions
        img_data = np.expand_dims(img_data, axis=(0))  # TODO only use this for blosc2?!

        # Apply normalization using the provided scheme
        normalized_np = apply_normalization(
            scheme=NormalizationScheme.Z_SCORE,
            image=img_data,
            target_dtype=np.float32,
            use_mask_for_norm=False,
        )

        img_tensor = torch.from_numpy(normalized_np).to(device)

        patch_size = tuple(config["recommended_downstream_patchsize"])

        try:
            print(f"Attempting to resize from {img_tensor.shape} to {patch_size}")
            if img_tensor.shape[-3:] != patch_size:
                img_tensor = F.interpolate(
                    img_tensor, size=patch_size, mode="trilinear", align_corners=False
                )
        except Exception as e:
            print(f"Error during resize: {str(e)}")
            raise

        # Generate embeddings
        with torch.no_grad():
            embeddings = model(img_tensor)

        return embeddings, affine

    except Exception as e:
        print(f"Error in process_image for {img_path}: {str(e)}")
        raise


def main(args):
    # Setup distributed processing
    rank, world_size, gpu = setup_distributed()

    if rank == 0:
        print(f"Using {world_size} GPUs for processing")

    processing_function_map = {
        "native": process_image_native_res,
        "resized": process_image,
        "center_crop": process_image_center_crop,
    }

    base_path = args.base_path
    output_base = args.output_base
    input_dirs = args.input_dirs
    process_image_fct = processing_function_map[args.processing_function]

    if rank == 0:
        print(f"Using processing function: {process_image_fct.__name__}")

    # Setup model on the specific GPU
    device = torch.device(f"cuda:{gpu}")

    model, config = resenc.setup_model(base_path)
    model = model.to(device)

    blosc2.set_nthreads(1)

    # Process each directory
    for input_dir in input_dirs:
        dir_name = os.path.basename(input_dir)
        output_dir = os.path.join(output_base, dir_name)

        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # Sync all processes
        if dist.is_initialized():
            dist.barrier()

        # Collect all image files
        img_files = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.endswith((".nii.gz", ".nrrd", ".b2nd")):
                    rel_path = os.path.relpath(os.path.join(root, f), input_dir)
                    img_files.append(rel_path)

        # we add this so that we can run multiple times the same script to utilize multiple nodes
        random.shuffle(img_files)

        # Distribute files across GPUs
        my_files = distribute_files(img_files, rank, world_size)

        if rank == 0:
            print(f"Total images: {len(img_files)}")
            print(f"GPU {rank} processing: {len(my_files)} files")

        # Process files assigned to this GPU
        for img_file in tqdm(my_files, desc=f"GPU {rank} - Processing {dir_name}"):
            output_path = Path(output_dir) / f"{Path(img_file).stem}_embedding.bl2"
            if output_path.exists():
                print(f"SKIPPING - Embedding already exist: {output_path}")
                continue

            img_path = os.path.join(input_dir, img_file)
            try:
                embeddings, _ = process_image_fct(img_path, model, config, device)

                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                for i, emb in enumerate(embeddings):
                    if i == 0:
                        # torch.Size([1, 640, 6, 6, 6])
                        save_embeddings_blosc2(emb, output_path)
                    else:
                        if i <= 2:
                            # torch.Size([1, 640, 12, 12, 12])
                            # torch.Size([1, 512, 24, 24, 24])
                            save_embeddings_blosc2(
                                emb,
                                Path(output_dir) / f"{Path(img_file).stem}_stage{i}_embedding.bl2"
                            )
                        else:
                            # We are only interested in the first three stages
                            continue

                print(f"GPU {rank} - Successfully processed {img_file}")

            except Exception as e:
                print(f"GPU {rank} - Error processing {img_file}: {str(e)}")
                continue

    # Clean up
    cleanup_distributed()


def parse_args():
    # # Define paths
    # base_path = "/p/scratch/hai_1023/rocket_vlm/vision_encoders/pretrained_primus_mae/Dataset807_Rocketv2_exclude_CTRATE_val/EvaMAETrainer2500Epochs__nnsslPlans__mask_ratio08__vit_patch_size888__embed_dim864__encoder_eva_depth28__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres"
    # output_base = "/home/r403k/Data/papertrack25/segmentations/test_embeddings"

    # input_dirs = [
    #     "/home/r403k/ResourcesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/imagesTr",
    #     "/home/r403k/ResourcesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/imagesVa"
    # ]
    # # Define function to use for embedding creation (native resolution or resized)
    # # Use process_image for resized images or process_image_native_res for native resolution
    # process_image_fct = process_image

    import argparse

    parser = argparse.ArgumentParser(
        description="Embed images using a pretrained model."
    )
    parser.add_argument(
        "--base_path",
        required=True,
        type=str,
        help="Path to the model directory containing plans.json and fold_all/checkpoint_final.pth",
    )
    parser.add_argument(
        "--output_base",
        required=True,
        type=str,
        help="Base directory to save the output embeddings (.npz files).",
    )
    parser.add_argument(
        "--input_dirs",
        required=True,
        nargs="+",
        type=str,
        help="List of input directories containing .nii.gz files.",
    )
    parser.add_argument(
        "--processing_function",
        required=True,
        type=str,
        choices=["native", "resized", "region", "center_crop"],
        help="Embedding generation method: 'native' (patch-based full image), 'resized' (resize full image), 'region' (patch-based region extraction).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
        print("Processing finished.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)  # Exit with error code
