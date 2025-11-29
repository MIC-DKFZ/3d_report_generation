# FIXME: Once we define which FeatureMaps we will use for creating the embeddings,
#  this whole file can be removed and the resenc method can be integrated in embedd_images_primus.py
#  with all the logic for the embeddings creation in the forward pass in resenc.py

from typing import List
import resenc
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import blosc2
import traceback
from einops import rearrange
import random
from pathlib import Path

from prep.normalization_schemes import (
    apply_normalization,
    NormalizationScheme,
)

import utils

import torch.distributed as dist
import os


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


def center_crop_or_pad_3d(x: torch.Tensor, target_size=(192, 192, 192)) -> torch.Tensor:
    """
    x: [B, C, H, W, D]
    target_size: (tH, tW, tD)
    Center-crop if image > target, center-pad with zeros if image < target.
    """
    assert x.ndim == 5, f"Expected 5D tensor [B, C, H, W, D], got {x.shape}"

    B, C, H, W, D = x.shape
    tH, tW, tD = target_size

    out = x.new_zeros((B, C, tH, tW, tD))

    def compute_indices(orig, target):
        """
        Returns (src_start, src_end, dst_start, dst_end) so that
        out[..., dst_start:dst_end] = x[..., src_start:src_end]
        is a centered crop (orig >= target) or centered pad (orig < target).
        """
        copy_len = min(orig, target)
        src_start = max((orig - target) // 2, 0)
        src_end = src_start + copy_len
        dst_start = max((target - orig) // 2, 0)
        dst_end = dst_start + copy_len
        return src_start, src_end, dst_start, dst_end

    h_s0, h_s1, h_d0, h_d1 = compute_indices(H, tH)
    w_s0, w_s1, w_d0, w_d1 = compute_indices(W, tW)
    d_s0, d_s1, d_d0, d_d1 = compute_indices(D, tD)

    out[:, :, h_d0:h_d1, w_d0:w_d1, d_d0:d_d1] = x[:, :, h_s0:h_s1, w_s0:w_s1, d_s0:d_s1]
    return out

def process_image_center_crop(
    img_path, model, config, device="cuda" if torch.cuda.is_available() else "cpu"
):
    target_size = (192, 192, 192)  # (H, W, D)
    try:
        blosc2_result = utils.load_blosc2(img_path)
        img_data, _ = blosc2_result
        img = np.array(img_data)

        # Add batch dim; if no channel dim, add one
        # -> expect final shape [B, C, H, W, D]
        if img.ndim == 3:
            # [H, W, D] -> [1, 1, H, W, D]
            img = np.expand_dims(img, axis=(0, 1))
        elif img.ndim == 4:
            # assume [C, H, W, D] -> [1, C, H, W, D]
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 5:
            # assume already [B, C, H, W, D]
            pass
        else:
            raise ValueError(f"Unexpected img ndim {img.ndim}, shape {img.shape}")

        normalized_np = apply_normalization(
            scheme=NormalizationScheme.Z_SCORE,
            image=img,
            target_dtype=np.float32,
            use_mask_for_norm=False,
        )

        img_tensor = torch.from_numpy(normalized_np).to(device)
        print(f"Tensor shape after adding batch/channel dims: {img_tensor.shape}")
        # assert shape is indeed [B, C, H, W, D]
        assert img_tensor.ndim == 5, f"Expected [B, C, H, W, D], got {img_tensor.shape}"

        # Center crop/pad
        img_tensor = center_crop_or_pad_3d(img_tensor, target_size)

        with torch.no_grad():
            embeddings = model(img_tensor)

        return embeddings, None

    except Exception as e:
        print(f"Error in process_image for {img_path}: {str(e)}")
        raise


@torch.inference_mode()
def process_image_native_res(
    img_path, model, config, device="cuda" if torch.cuda.is_available() else "cpu"
):
    raise Exception("attentionmask not supported in current Primus implementation")
    # Print debug information about patch size
    patch_size = tuple(config["recommended_downstream_patchsize"])
    try:
        blosc2_result = utils.load_blosc2(img_path)
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


def process_image(
    img_path, model, config, device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Print debug information about patch size
    patch_size = tuple(config["recommended_downstream_patchsize"])
    # print(f"Target patch size from config: {patch_size}")
    try:
        img_data, affine = utils.load_image_data_and_affine(
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
    rank, world_size, gpu = utils.setup_distributed()

    if rank == 0:
        print(f"Using {world_size} GPUs for processing")

    processing_function_map = {
        "native": process_image_native_res,
        "resized": process_image,
        "center_crop": process_image_center_crop,
    }

    checkpoint_path = args.checkpoint_path
    output_base = args.output_base
    input_dirs = args.input_dirs
    process_image_fct = processing_function_map[args.processing_function]

    if rank == 0:
        print(f"Using processing function: {process_image_fct.__name__}")

    # Setup model on the specific GPU
    device = torch.device(f"cuda:{gpu}")

    model, config = resenc.setup_model(checkpoint_path)
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

        # Distribute files across GPUs
        my_files = utils.distribute_files(img_files, rank, world_size)

        if rank == 0:
            print(f"Total images: {len(img_files)}")
            print(f"GPU {rank} processing: {len(my_files)} files")

        # Process files assigned to this GPU
        for img_file in tqdm(my_files, desc=f"GPU {rank} - Processing {dir_name}"):
            output_path = (
                Path(output_dir) / f"{Path(img_file.replace('nii.gz', '')).stem}_embedding.bl2"
            )
            p1 = output_path
            p2 = Path(str(output_path).replace("_embedding.bl2", "_stage1_embedding.bl2"))
            p3 = Path(str(output_path).replace("_embedding.bl2", "_stage2_embedding.bl2"))
            if p1.exists() and p2.exists() and p3.exists():
                try:
                    blosc2.load_array(p1)
                    blosc2.load_array(p2)
                    blosc2.load_array(p3)
                    print(f"SKIPPING - Embedding already exist: {output_path}")
                    continue
                except Exception:
                    pass

            img_path = os.path.join(input_dir, img_file)
            try:
                embeddings, _ = process_image_fct(img_path, model, config, device)

                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                for i, emb in enumerate(embeddings):
                    if i == 0:
                        # torch.Size([1, 640, 6, 6, 6])
                        utils.save_embeddings_blosc2(emb, output_path)
                    else:
                        if i <= 2:
                            # torch.Size([1, 640, 12, 12, 12])
                            # torch.Size([1, 512, 24, 24, 24])
                            utils.save_embeddings_blosc2(
                                emb,
                                Path(output_dir)
                                / f"{Path(img_file.replace('nii.gz', '')).stem}_stage{i}_embedding.bl2",
                            )
                        else:
                            # We are only interested in the first three stages
                            continue

                print(f"GPU {rank} - Successfully processed {img_file}")

            except Exception as e:
                print(f"GPU {rank} - Error processing {img_file}: {str(e)}")
                continue

    # Clean up
    utils.cleanup_distributed()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Embed images using a pretrained model.")
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to the model checkpoint.pth",
    )
    parser.add_argument(
        "--output_base",
        required=True,
        type=str,
        help="Base directory to save the output embeddings (.bl2 files).",
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
        choices=["native", "resized", "center_crop"],
        help="Embedding generation method: 'native' (patch-based full image), 'resized' (resize full image).",
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
