from typing import List
from evavit import PretrainedEvaEncFromEvaMAE, get_config_from_plans
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

from einops import rearrange

from nnssl.preprocessing.normalization.normalization_schemes import apply_normalization, NormalizationScheme

def setup_model(base_path):
    plans_file = f"{base_path}/plans.json"
    checkpoint_path = f"{base_path}/fold_all/checkpoint_final.pth"

    config = get_config_from_plans(plans_file)
    model = PretrainedEvaEncFromEvaMAE(
        input_channels=1,
        output_channels=1,
        pretrained_weights_file=checkpoint_path,
        **config
    )
    model.eval()
    return model, config


def split_image_into_patches(img, patch_size: List[int], vit_patch_size: List[int], foreground_mask: torch.Tensor=None):
    # Get image shape
    img_shape = img.shape[-3:]  # Shape should be (B, C, H, W, D) -> (H, W, D)

    # Calculate number of patches in each dimension
    num_patches = [int(np.ceil(s / p)) for s, p in zip(img_shape, patch_size)]

    attn_mask = torch.ones([nps * int(ps / vps) for nps, ps, vps in zip(num_patches, patch_size, vit_patch_size)], dtype=torch.bool)

    # Calculate padding for each dimension
    padding = [((p - s) % p) for s, p in zip(img_shape, patch_size)]
    tokens_to_mask = [int(np.floor(pad / vps)) for pad, vps in zip(padding, vit_patch_size)]
    if tokens_to_mask[0] > 0:
        attn_mask[-tokens_to_mask[0]:,:,:] = False
    if tokens_to_mask[1] > 0:
        attn_mask[:,-tokens_to_mask[1]:,:] = False
    if tokens_to_mask[2] > 0:
        attn_mask[:,:,-tokens_to_mask[2]:] = False

    # Pad image to fit patches
    img_padded = F.pad(img, tuple([x for xs in [[0, p] for p in padding[::-1]] for x in xs]), mode='constant')

    # Split image into patches
    patches = rearrange(img_padded, 'B C (m x) (n y) (o z) -> B C (m n o) x y z', m=num_patches[0], n=num_patches[1], o=num_patches[2], x=patch_size[0], y=patch_size[1], z=patch_size[2])

    # Split attention mask into patches
    reduced_ps = [int(p / vps) for p, vps in zip(patch_size, vit_patch_size)]
    attn_mask = rearrange(attn_mask, '(m x) (n y) (o z) -> (m n o) x y z', m=num_patches[0], n=num_patches[1], o=num_patches[2], x=reduced_ps[0], y=reduced_ps[1], z=reduced_ps[2])

    if foreground_mask is not None:
        # Pad foreground mask to fit patches
        foreground_mask_padded = F.pad(foreground_mask, tuple([x for xs in [[0, p] for p in padding[::-1]] for x in xs]), mode='constant')
        # This creates a 24x24x24 grid of patches, each size 8x8x8 (if vit_patch_size = 8 and patch_size / vit_patch_size = 24)
        foreground_mask_patches = rearrange(foreground_mask_padded, '(m x) (n y) (o z) -> (m n o) x y z', x=vit_patch_size[0], y=vit_patch_size[1], z=vit_patch_size[2])
        # If any voxel in a 8x8x8 patch contains foreground, set the whole patch to foreground
        # (24 * 24 * 24, 8, 8, 8) -> (24, 24, 24)
        mask = torch.any(foreground_mask_patches != 0, dim=(-3, -2, -1))
        #
        foreground_mask_for_mask = rearrange(mask, '(m n o) -> m n o', m=reduced_ps[0], n=reduced_ps[1], o=reduced_ps[2]).unsqueeze(0)
        attn_mask = attn_mask | foreground_mask_for_mask.to(attn_mask.device)

    # (B, C, N, H, W, D) -> (N, B, C, H, W, D)
    return patches.permute(-4, 0, 1, -3, -2, -1), attn_mask, num_patches


def combine_patches_into_image(embeddings, attn_masks, num_patches):
    whole_image_with_padding = rearrange(torch.stack(embeddings), '(X Y Z) B C H W D -> B C (X H) (Y W) (Z D)', X=num_patches[0], Y=num_patches[1], Z=num_patches[2])
    whole_attention_mask = rearrange(attn_masks, '(X Y Z) H W D -> (X H) (Y W) (Z D)', X=num_patches[0], Y=num_patches[1], Z=num_patches[2])
    true_indices = torch.nonzero(whole_attention_mask)
    min_indices = true_indices.min(dim=0).values
    max_indices = true_indices.max(dim=0).values
    # + 1 since slice end is exclusive
    whole_image_cropped = whole_image_with_padding[:, :, min_indices[0]:max_indices[0] + 1, min_indices[1]:max_indices[1] + 1, min_indices[2]:max_indices[2] + 1]
    return whole_image_cropped


def load_yaml(file_path: str):
    try:
        with open(file_path, 'r') as f:
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
    rocket2raw = {d['image_id']: Path(d['image_filepaths'][0]).stem.split('.')[0] for d in rocket_data}
    raw2rocket = {v: k for k, v in rocket2raw.items()}
    return rocket2raw, raw2rocket


def get_seg_path_from_img_path(img_path: str, raw2rocket: dict):
    img_name = Path(img_path).name
    if img_name.endswith(".nii.gz"):
        img_name = img_name[:-7]
    else:
        img_name = Path(img_name).stem

    if img_name.endswith("_0000"):
        img_name = img_name[:-5]
    rocket_name = raw2rocket[img_name]
    # seg_path = f"/home/r403k/ResourcesDKFZ/cluster_data/r403k/papertrack25/YannickResEnc/amos-mm/{rocket_name}"
    segmentation_path = f"/home/r403k/ResourcesDKFZ/cluster_data/r403k/papertrack25/YannickResEnc/amos-mm/chunk_0/{rocket_name}.nrrd"
    return segmentation_path

def process_region_from_segmentation_native_res(
        img_path,
        model,
        config,
        device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Print debug information about patch size
    # TODO: How to handle segmentation_path?
    segmentation_path = "/home/r403k/ResourcesDKFZ/cluster_data/r403k/papertrack25/YannickResEnc/amos-mm/chunk_0"
    # heart: 28, kidney_left: 42, kidney_right: 43, liver: 44,
    idx_class_name_mapping = json.load(open("/home/r403k/ResourcesDKFZ/cluster_checkpoints/y033f/benchmarkability_trained_models/Dataset351_TotalSegmentator_v2/nnUNetTrainer_2000epochs_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres/dataset.json"))
    patch_size = tuple(config['input_shape'])
    print(f"Target patch size from config: {patch_size}")
    rocket2raw, raw2rocket = _create_name_mapping_rocket_raw()
    try:
        img = nib.load(img_path)
        seg, header = nrrd.read(get_seg_path_from_img_path(img_path, raw2rocket))
        img_data = img.get_fdata()
        seg_data = seg
        mask_idx_list = np.unique(seg_data)
        set_of_interesting_masks = [28, 42, 43, 44]
        # Add batch and channel dimensions
        img_data = np.expand_dims(img_data, axis=(0, 1))
        affine = img.affine

        print(f"Original image shape: {img_data.shape}")

        # Apply normalization using the provided scheme
        normalized_np = apply_normalization(
            scheme=NormalizationScheme.Z_SCORE,
            image=img_data,
            target_dtype=np.float32,
            use_mask_for_norm=False,
        )
        for mask_idx in mask_idx_list:
            if mask_idx in set_of_interesting_masks:
                # TODO: Add feature where one can specify the mask indicies to use (i.e. lists which combine indicies, e.g. to have the left lung lobes combined.)
                # matching_indicies_x = np.where(np.isin(seg_data, mask_idx))[0]
                # if len(matching_indicies_x) > 0:
                #     x_min, x_max = matching_indicies_x.min(), matching_indicies_x.max()
                x_min, x_max = np.where(seg_data == mask_idx)[0].min(), np.where(seg_data == mask_idx)[0].max()
                y_min, y_max = np.where(seg_data == mask_idx)[1].min(), np.where(seg_data == mask_idx)[1].max()
                z_min, z_max = np.where(seg_data == mask_idx)[2].min(), np.where(seg_data == mask_idx)[2].max()

                # normalized_np: [B, C, H, W, D]
                img_data_cropped = normalized_np[: ,: , x_min:x_max, y_min:y_max, z_min:z_max]
                # seg_data [H, W, D]
                seg_data_cropped = seg_data[x_min:x_max, y_min:y_max, z_min:z_max]
                # set mask to with mask_idx to 1 and all other to 0
                seg_data_cropped = np.where(seg_data_cropped == mask_idx, 1, 0)

                # Convert back to tensor
                img_tensor = torch.from_numpy(img_data_cropped).to(device)
                print(f"Tensor shape after adding batch/channel dims: {img_tensor.shape}")

                # Get patch size from config and ensure it's a tuple
                patch_size = tuple(config['input_shape'])  # Should be (192, 192, 192)
                # FIXME: currenlty function only supports Batch size of 1
                patches, attn_masks, num_patches = split_image_into_patches(
                    img_tensor, patch_size, vit_patch_size=[8, 8, 8],
                    foreground_mask=torch.from_numpy(seg_data_cropped).to(device))

                embeddings = []
                # Generate embeddings
                for patch, attn_mask in zip(patches, attn_masks):
                    with torch.no_grad():
                        patch_embeddings = model(patch, rearrange(attn_mask, 'h w d -> (h w d)').to(device))
                    embeddings.append(patch_embeddings)
                embeddings = combine_patches_into_image(embeddings, attn_masks.to(device), num_patches)

        return embeddings.cpu(), affine

    except Exception as e:
        print(f"Error in process_image for {img_path}: {str(e)}")
        raise


def process_image_native_res(img_path, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Print debug information about patch size
    patch_size = tuple(config['input_shape'])
    print(f"Target patch size from config: {patch_size}")
    try:
        img = nib.load(img_path)
        img_data = img.get_fdata()
        # Add batch and channel dimensions
        img_data = np.expand_dims(img_data, axis=(0, 1))
        affine = img.affine

        print(f"Original image shape: {img_data.shape}")

        # Apply normalization using the provided scheme
        normalized_np = apply_normalization(
            scheme=NormalizationScheme.Z_SCORE,
            image=img_data,
            target_dtype=np.float32,
            use_mask_for_norm=False,
        )

        # Convert back to tensor
        img_tensor = torch.from_numpy(normalized_np).to(device)
        print(f"Tensor shape after adding batch/channel dims: {img_tensor.shape}")


        # Get patch size from config and ensure it's a tuple
        patch_size = tuple(config['input_shape'])  # Should be (192, 192, 192)
        # FIXME: currenlty function only supports Batch size of 1
        patches, attn_masks, num_patches = split_image_into_patches(img_tensor, patch_size, vit_patch_size=[8, 8, 8])

        # Resize image to match patch size with additional checks

        embeddings = []
        # Generate embeddings
        for patch, attn_mask in zip(patches, attn_masks):
            with torch.no_grad():
                patch_embeddings = model(patch, rearrange(attn_mask, 'h w d -> (h w d)').to(device))
            embeddings.append(patch_embeddings)
        embeddings = combine_patches_into_image(embeddings, attn_masks.to(device), num_patches)

        return embeddings.cpu(), affine

    except Exception as e:
        print(f"Error in process_image for {img_path}: {str(e)}")
        raise


def process_image(img_path, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Print debug information about patch size
    patch_size = tuple(config['input_shape'])
    print(f"Target patch size from config: {patch_size}")
    try:
        img = nib.load(img_path)
        img_data = img.get_fdata()
        # Add batch and channel dimensions
        img_data = np.expand_dims(img_data, axis=(0, 1))
        affine = img.affine

        print(f"Original image shape: {img_data.shape}")

        # Apply normalization using the provided scheme
        normalized_np = apply_normalization(
            scheme=NormalizationScheme.Z_SCORE,
            image=img_data,
            target_dtype=np.float32,
            use_mask_for_norm=False,
        )

        # Convert back to tensor
        img_tensor = torch.from_numpy(normalized_np).to(device)
        print(f"Tensor shape after adding batch/channel dims: {img_tensor.shape}")


        # Get patch size from config and ensure it's a tuple
        patch_size = tuple(config['input_shape'])  # Should be (192, 192, 192)

        # Resize image to match patch size with additional checks
        try:
            print(f"Attempting to resize from {img_tensor.shape} to {patch_size}")
            if img_tensor.shape[-3:] != patch_size:
                img_tensor = F.interpolate(
                    img_tensor,
                    size=patch_size,
                    mode='trilinear',
                    align_corners=False
                )
            else:
                print("Image already at target size, skipping resize")
        except Exception as e:
            print(f"Error during resize: {str(e)}")
            raise
        print(f"Input shape after resizing: {img_tensor.shape}")

        # Generate embeddings
        with torch.no_grad():
            embeddings = model(img_tensor)

        return embeddings.cpu(), affine

    except Exception as e:
        print(f"Error in process_image for {img_path}: {str(e)}")
        raise

def main():
    # Define paths
    base_path = "/home/r403k/ResourcesDKFZ/e230-thrp-data/mic_rocket/checkpoints/pretraining/MAE/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres"
    output_base = "/home/r403k/ResourcesDKFZ/E132-Projekte/Projects/2025_MIC_Rocket_VLM/vision_embeddings/192_sliding_window/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres"

    input_dirs = [
        "/home/r403k/ResourcesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/imagesTr",
        "/home/r403k/ResourcesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/imagesVa"
    ]

    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = setup_model(base_path)
    model = model.to(device)

    # Define function to use for embedding creation (native resolution or resized)
    # Use process_image for resized images or process_image_native_res for native resolution
    process_image_fct = process_region_from_segmentation_native_res

    # Process each directory
    for input_dir in input_dirs:
        dir_name = os.path.basename(input_dir)
        output_dir = os.path.join(output_base, dir_name)
        os.makedirs(output_dir, exist_ok=True)

        # Get list of files first
        img_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]

        # Process files with tqdm
        for img_file in tqdm(img_files, desc=f"Processing {dir_name}"):
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file.replace('.nii.gz', '_embedding.npz'))

            if os.path.exists(output_path):
                print(f"Skipping {img_file} - embedding already exists")
                continue

            try:
                embeddings, affine = process_image_fct(img_path, model, config, device)

                # Print embedding information
                print(f"Embedding shape: {embeddings.shape}")
                print(f"Value range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
                print(f"Mean: {embeddings.mean():.3f}")
                print(f"Std: {embeddings.std():.3f}")


                # Save embeddings and metadata
                np.savez_compressed(
                    output_path,
                    embeddings=embeddings.numpy(),
                    affine=affine,
                    original_file=img_file
                )

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue  # Continue with next file instead of stopping

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")