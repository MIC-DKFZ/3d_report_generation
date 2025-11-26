import colipri
import torch
import os
import numpy as np
from tqdm import tqdm
import blosc2
import traceback
from pathlib import Path

import utils

import torch.distributed as dist
import os


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
    target_size = (160, 160, 160)  # (H, W, D)
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

        # FIXME: Colipri expects a very specific preprocessing, which we already applied before
        # normalized_np = apply_normalization(
        #     scheme=NormalizationScheme.Z_SCORE,
        #     image=img,
        #     target_dtype=np.float32,
        #     use_mask_for_norm=False,
        # )

        img_tensor = torch.from_numpy(img).to(device)
        print(f"Tensor shape after adding batch/channel dims: {img_tensor.shape}")
        # assert shape is indeed [B, C, H, W, D]
        assert img_tensor.ndim == 5, f"Expected [B, C, H, W, D], got {img_tensor.shape}"

        # Center crop/pad
        img_tensor = center_crop_or_pad_3d(img_tensor, target_size)

        with torch.no_grad():
            # we actually get two embeddings (image embeddign + langauge algined ones)
            embeddings = model(img_tensor)

        return embeddings, None

    except Exception as e:
        print(f"Error in process_image for {img_path}: {str(e)}")
        raise


def main(args):
    # Setup distributed processing
    rank, world_size, gpu = utils.setup_distributed()

    if rank == 0:
        print(f"Using {world_size} GPUs for processing")

    processing_function_map = {
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

    model, config = colipri.setup_model(checkpoint_path)
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
                Path(output_dir)
                / f"{Path(img_file.replace('nii.gz', '')).stem}_embedding.bl2"
            )
            if os.path.exists(output_path):
                try:
                    blosc2.load_array(output_path)
                    print(f"SKIPPING - Embedding already exist: {output_path}")
                    continue
                except Exception:
                    pass

            img_path = os.path.join(input_dir, img_file)
            try:
                _embeddings, affine = process_image_fct(img_path, model, config, device)

                embeddings, pooling_output = _embeddings

                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                utils.save_embeddings_blosc2(embeddings, output_path)
                utils.save_embeddings_blosc2(pooling_output, Path(str(output_path).replace('_embedding.bl2', '_poolingembedding.bl2')))
                print(f"GPU {rank} - Successfully processed {img_file}")

            except Exception as e:
                print(f"GPU {rank} - Error processing {img_file}: {str(e)}")
                continue

    # Clean up
    utils.cleanup_distributed()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Embed images using a pretrained model."
    )
    parser.add_argument(
        "--checkpoint_path",
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
