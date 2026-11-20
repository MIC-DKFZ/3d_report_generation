"""Reorientation, preprocessing and encoding of NIfTI volumes for inference on new images."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import blosc2
import numpy as np
import pytest
import SimpleITK as sitk
import torch
from dynamic_network_architectures.architectures.primus import Primus

import inference_nifti
from vlm.configs.data_config import (
    DataConfig,
    VisionInputPreprocessingConfig,
)
from vlm.data.preprocessing import nifti_inputs
from vlm.data.preprocessing.encoder_input import (
    preprocess_encoder_input,
    resize_with_anti_aliasing,
)
from vlm.data.vision_input_settings import load_settings
from vlm.models.vision_encoders import nnssl, primus
from vlm.models.vision_encoders.primus import PrimusEncoder
from vlm.scripts.embeddings import generate_embeddings

# A tiny Primus, in the plan format of nnssl's PrimusX.
TINY_ARCH_KWARGS = {
    "input_channels": 1,
    "embed_dim": 24,
    "patch_embed_size": [4, 4, 4],
    "output_channels": 1,
    "input_shape": [16, 16, 16],
    "encoder_eva_depth": 2,
    "encoder_eva_numheads": 2,
    "decoder_eva_depth": 1,
    "decoder_eva_numheads": 2,
    "patch_drop_rate": 0.8,
    "drop_path_rate": 0.2,
    "init_values": 0.1,
    "scale_attn_inner": True,
}


# A tiny ResEnc U-Net, in the plan format of nnssl's 'ResidualEncoderUNet'.


def tiny_checkpoint(tmp_path, spacing_style="noresample", spacing=None, downstream_size=16):
    torch.manual_seed(0)
    network = Primus(
        1, 24, (4, 4, 4), 1, 2, 2, (16, 16, 16), init_values=0.1, scale_attn_inner=True
    )
    weights = network.state_dict()
    # The pre-training decoder in the checkpoint is not part of the encoder.
    weights["decoder.norm.weight"] = torch.ones(24)
    architecture_plans = {"arch_class_name": "PrimusX", "arch_kwargs": TINY_ARCH_KWARGS}
    return save_nnssl_checkpoint(
        tmp_path,
        weights,
        architecture_plans,
        ("down_projection", "eva"),
        spacing_style,
        spacing,
        key_to_lpe="eva.pos_embed",
        downstream_size=downstream_size,
    )


def save_nnssl_checkpoint(
    tmp_path,
    weights,
    architecture_plans,
    stem_and_encoder,
    spacing_style="noresample",
    spacing=None,
    key_to_lpe=None,
    downstream_size=16,
):
    path = tmp_path / "checkpoint_final.pth"
    plan = {
        "architecture_plans": architecture_plans,
        "pretrain_plan": {
            "configurations": {
                spacing_style: {
                    "spacing_style": spacing_style,
                    "spacing": spacing,
                    "normalization_schemes": ["ZScoreNormalization"],
                    "patch_size": [16, 16, 16],
                }
            }
        },
        "pretrain_num_input_channels": 1,
        "recommended_downstream_patchsize": (downstream_size,) * 3,
        "key_to_stem": stem_and_encoder[0],
        "key_to_encoder": stem_and_encoder[1],
        "key_to_lpe": key_to_lpe,
    }
    torch.save({"network_weights": weights, "nnssl_adaptation_plan": plan}, path)
    return path, weights


def test_positional_embedding_is_resized_to_the_recommended_patch_size(tmp_path):
    path, weights = tiny_checkpoint(tmp_path, downstream_size=24)

    encoder = PrimusEncoder.from_pretrained(path)

    assert encoder.patch_size == (24, 24, 24)
    torch.testing.assert_close(
        encoder.eva.pos_embed,
        nnssl.resize_pos_embed(weights["eva.pos_embed"], (4, 4, 4), (6, 6, 6)),
    )
    with torch.no_grad():
        assert encoder(torch.randn(1, 1, 24, 24, 24)).shape == (1, 24, 6, 6, 6)


def test_padded_attention_head_dim_matches_unpadded_attention(tmp_path, monkeypatch):
    # Head dim 12 is padded to 16; the result must not change.
    sdpa = torch.nn.functional.scaled_dot_product_attention
    encoder = PrimusEncoder.from_pretrained(tiny_checkpoint(tmp_path)[0])
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        padded = encoder(x)
        assert torch.nn.functional.scaled_dot_product_attention is sdpa
        monkeypatch.setattr(primus, "padded_attention_head_dim", contextlib.nullcontext)
        unpadded = encoder(x)

    torch.testing.assert_close(padded, unpadded)


def write_volume(path, array, direction, spacing=(1.0, 1.0, 1.0)):
    image = sitk.GetImageFromArray(array)
    image.SetDirection(direction)
    image.SetSpacing(spacing)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def make_dataset(input_dir, mask_dir=None, orientation="LPS", **kwargs):
    preprocessing = {
        "target_spacing": (1.0, 1.0, 1.0),
        "normalization": "z_score",
        "input_shape": (8, 8, 8),
        "processing_mode": "resized",
    }
    files = nifti_inputs.find_nifti_files(input_dir)
    return nifti_inputs.NiftiEncoderInputDataset(
        input_dir, files, orientation, mask_dir, preprocessing, **kwargs
    )


def test_volumes_are_reoriented_before_preprocessing(tmp_path):
    rng = np.random.default_rng(0)
    array = rng.normal(size=(12, 10, 14)).astype(np.float32)
    lps = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    las = (1, 0, 0, 0, -1, 0, 0, 0, 1)
    # The same patient stored in LPS and in LAS (anterior-posterior axis flipped).
    write_volume(tmp_path / "lps/scan.nii.gz", array, lps)
    write_volume(tmp_path / "las/scan.nii.gz", array[:, ::-1].copy(), las)

    from_lps = make_dataset(tmp_path / "lps")[0]["encoder_input"]
    from_las = make_dataset(tmp_path / "las")[0]["encoder_input"]
    from_las_kept = make_dataset(tmp_path / "las", orientation="LAS")[0]["encoder_input"]

    assert from_lps.shape == (1, 8, 8, 8)
    torch.testing.assert_close(from_las, from_lps)
    torch.testing.assert_close(from_las_kept, from_lps.flip(-2))


def inference_args(**overrides):
    args = dict.fromkeys(inference_nifti.PREPROCESSING_FIELDS)
    return SimpleNamespace(**{**args, "mask_dir": None, **overrides})


ENCODER = SimpleNamespace(target_spacing=(2.0, 1.0, 1.0), normalization="z_score")
CONFIGURED = VisionInputPreprocessingConfig(
    processing_mode="center_crop", roi_mask="lungs", mask_label_ids=[1], target_spacing=None
)


def test_preprocessing_comes_from_the_run_config_and_arguments_override_it(tmp_path):
    args = inference_args(mask_dir=tmp_path, outlier_min_voxels=10)

    preprocessing, sources = inference_nifti.resolve_preprocessing(args, CONFIGURED, ENCODER)

    # The run config's 'target_spacing: null' means no resampling, not the encoder's spacing.
    assert preprocessing.target_spacing is None and sources["target_spacing"] == "run config"
    assert preprocessing.processing_mode == "center_crop"
    assert preprocessing.mask_label_ids == [1] and preprocessing.roi_mask == "lungs"
    assert preprocessing.outlier_min_voxels == 10 and sources["outlier_min_voxels"] == "argument"


@pytest.mark.parametrize(
    "embedding, roi_mask",
    [
        ("nnFoundationViT", "TotalSegmentator 'total', lungs (labels 10-14)"),
        ("Primus-1B_Whole", None),
    ],
)
def test_public_script_saves_the_paper_embeddings_with_their_settings(
    tmp_path, monkeypatch, embedding, roi_mask
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint, _ = tiny_checkpoint(tmp_path)
    rng = np.random.default_rng(0)
    mask = np.zeros((24, 22, 26), np.uint8)
    mask[3:20, 4:18, 2:12] = 10  # a lung lobe
    mask[3:20, 4:18, 14:24] = 1  # another organ
    ras = (-1, 0, 0, 0, -1, 0, 0, 0, 1)
    names = ("train_fixed/a/x_1", "valid_fixed/b/y_1")
    for name in names:
        volume = rng.normal(size=(24, 22, 26)).astype(np.float32)
        write_volume(tmp_path / f"in/{name}.nii.gz", volume, ras, spacing=(1.2, 1.0, 1.5))
        write_volume(tmp_path / f"masks/{name}.nii.gz", mask, ras, spacing=(1.2, 1.0, 1.5))
    mask_args = ("--mask_dir", str(tmp_path / "masks")) if roi_mask else ()

    generate_embeddings.main(
        [
            *("--embedding", embedding, "--dataset", "ctrate"),
            *("--input_dir", str(tmp_path / "in"), *mask_args),
            *("--output_dir", str(tmp_path / "out")),
            *("--checkpoint", str(checkpoint), "--num_workers", "0"),
            *("--save_workers", "0"),
        ]
    )

    settings = load_settings(tmp_path / "out")
    data = DataConfig(dataset_name="ctrate", answer_types="all", **settings.model_dump())
    assert data.vision_input_orientation == "LPS"
    assert data.embedding_encoder.model_dump() == {
        "class_name": "PrimusEncoder",
        "checkpoint_path": str(checkpoint),
    }
    preprocessing = data.vision_input_preprocessing
    assert preprocessing.roi_mask == roi_mask
    encoder = PrimusEncoder.from_pretrained(checkpoint)
    for name in names:
        mask_image = None
        if roi_mask:
            mask_image = sitk.DICOMOrient(sitk.ReadImage(tmp_path / f"masks/{name}.nii.gz"), "LPS")
        expected = encoder(
            preprocess_encoder_input(
                sitk.DICOMOrient(sitk.ReadImage(tmp_path / f"in/{name}.nii.gz"), "LPS"),
                mask_image,
                **preprocessing.preprocessing_kwargs(encoder.patch_size),
            )
        )[0].detach()
        saved = blosc2.load_array(
            str(tmp_path / "out" / Path(name).parts[0] / f"{Path(name).name}_embedding.bl2")
        )
        assert saved.shape == (24, 4, 4, 4)
        torch.testing.assert_close(torch.from_numpy(saved), expected)


def test_resizing_with_anti_aliasing_averages_detail_finer_than_the_output():
    # Stripes one voxel wide along x, shrunk by 64 / 15 (about CT-RATE's in-plane factor): they
    # should average to grey; trilinear resizing alone turns them into a coarser false pattern.
    stripes = torch.zeros(1, 1, 8, 8, 64)
    stripes[..., ::2] = 1.0
    plain = torch.nn.functional.interpolate(
        stripes, size=(8, 8, 15), mode="trilinear", align_corners=False
    )
    filtered = resize_with_anti_aliasing(stripes, (8, 8, 15))
    assert filtered.shape == (1, 1, 8, 8, 15)
    # Away from the borders, where the filter repeats the edge voxels.
    inner, plain_inner = filtered[..., 1:-1], plain[..., 1:-1]
    assert inner.std() < 0.01 and plain_inner.std() > 0.2
    torch.testing.assert_close(inner.mean(), torch.tensor(0.5), atol=0.01, rtol=0)


def test_direct_resizing_crops_the_same_roi_as_resampling_first():
    spacing_xyz = (0.5, 0.5, 2.0)  # SimpleITK order (x, y, z)
    array = np.zeros((20, 80, 80), np.float32)
    array[5:15, 20:60, 30:50] = 1.0  # the bright ROI, 20 x 20 x 10 mm
    mask = (array > 0).astype(np.uint8)
    image, mask_image = sitk.GetImageFromArray(array), sitk.GetImageFromArray(mask)
    for item in (image, mask_image):
        item.SetSpacing(spacing_xyz)
    kwargs = {
        "target_spacing": (1.0, 1.0, 1.0),
        "normalization": "none",
        "input_shape": (8, 8, 8),
        "enforce_min_bbox": False,
    }
    direct = preprocess_encoder_input(
        image, mask_image, processing_mode="resized_direct", **kwargs
    )
    resampled = preprocess_encoder_input(image, mask_image, processing_mode="resized", **kwargs)
    assert direct.shape == resampled.shape == (1, 1, 8, 8, 8)
    # Both crops hold mostly the ROI; its intensity fills the centre of the input.
    assert direct[..., 2:6, 2:6, 2:6].mean() > 0.9
    assert resampled[..., 2:6, 2:6, 2:6].mean() > 0.9
