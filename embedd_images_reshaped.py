from evavit import PretrainedEvaEncFromEvaMAE, get_config_from_plans
import nibabel as nib
import torch
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
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
    base_path = "/mnt/e230-thrp-data/mic_rocket/checkpoints/pretraining/MAE/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres"
    output_base = "/home/m340g/DrivesDKFZ/E132-Projekte/Projects/2025_MIC_Rocket_VLM/vision_embeddings/192_reshaped/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/2024_MICCAI24_AMOS-MM"
    
    input_dirs = [
        "/home/m340g/DrivesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/imagesTr",
        "/home/m340g/DrivesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/imagesVa"
    ]
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = setup_model(base_path)
    model = model.to(device)
    
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
                embeddings, affine = process_image(img_path, model, config, device)
                
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