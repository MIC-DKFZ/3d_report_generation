"""Reorientation, preprocessing and encoding of NIfTI volumes for inference on new images."""

import contextlib

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from dynamic_network_architectures.architectures.primus import Primus
from pydantic import ValidationError

from vlm.configs.data_config import DataConfig
from vlm.data.preprocessing import nifti_inputs
from vlm.models.vision_encoders import nnssl, primus, resenc
from vlm.models.vision_encoders.primus import PrimusEncoder
from vlm.models.vision_encoders.resenc import ResEncEncoder

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


@pytest.mark.parametrize("orientation", ["lps", "LAS", "RAI", "SPL"])
def test_orientation_codes_are_validated_and_upper_cased(orientation):
    data = DataConfig(
        dataset_name="ctrate", answer_types="all", vision_input_orientation=orientation
    )
    assert data.vision_input_orientation == orientation.upper()


@pytest.mark.parametrize("orientation", ["LP", "LPSI", "LLS", "LRS", "XYZ"])
def test_invalid_orientation_codes_are_rejected(orientation):
    with pytest.raises(ValidationError, match="vision_input_orientation"):
        DataConfig(dataset_name="ctrate", answer_types="all", vision_input_orientation=orientation)


# A tiny ResEnc U-Net, in the plan format of nnssl's 'ResidualEncoderUNet'.
TINY_RESENC_PLANS = {
    "arch_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
    "arch_kwargs": {
        "n_stages": 3,
        "features_per_stage": [4, 8, 16],
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[3, 3, 3]] * 3,
        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
        "n_blocks_per_stage": [1, 1, 2],
        "n_conv_per_stage_decoder": [1, 1],
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": True},
    },
    "arch_kwargs_requiring_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
}


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


def tiny_resenc_checkpoint(tmp_path):
    torch.manual_seed(0)
    weights = resenc.build_resenc(TINY_RESENC_PLANS, 1).state_dict()
    return save_nnssl_checkpoint(
        tmp_path, weights, TINY_RESENC_PLANS, ("encoder.stem", "encoder.stages")
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


def test_encoder_is_built_from_the_plan_of_an_nnssl_checkpoint(tmp_path):
    path, weights = tiny_checkpoint(tmp_path)

    encoder = PrimusEncoder.from_pretrained(path)

    assert not encoder.training
    assert encoder.embed_dim == 24
    assert encoder.downsample_factor == (4, 4, 4)
    assert encoder.patch_size == (16, 16, 16)
    assert encoder.target_spacing is None and encoder.normalization == "z_score"
    assert set(encoder.state_dict()) == {k for k in weights if k.startswith(("down_", "eva."))}
    torch.testing.assert_close(encoder.eva.pos_embed, weights["eva.pos_embed"])
    with torch.no_grad():
        assert encoder(torch.randn(2, 1, 16, 16, 16)).shape == (2, 24, 4, 4, 4)


def test_encoder_resamples_to_the_spacing_of_its_pre_training(tmp_path):
    path, _ = tiny_checkpoint(tmp_path, spacing_style="median", spacing=[2.0, 1.0, 1.0])

    assert PrimusEncoder.from_pretrained(path).target_spacing == (2.0, 1.0, 1.0)


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


def test_resize_pos_embed_keeps_prefix_tokens_and_an_unchanged_grid():
    pos_embed = torch.randn(1, 2 + 4**3, 8)
    resized = nnssl.resize_pos_embed(pos_embed, (4, 4, 4), (2, 4, 6))
    assert resized.shape == (1, 2 + 2 * 4 * 6, 8)
    torch.testing.assert_close(resized[:, :2], pos_embed[:, :2])
    torch.testing.assert_close(nnssl.resize_pos_embed(pos_embed, (4, 4, 4), (4, 4, 4)), pos_embed)


@pytest.mark.parametrize("encoder_class", [PrimusEncoder, ResEncEncoder])
def test_encoder_needs_a_checkpoint(encoder_class):
    with pytest.raises(ValueError, match="checkpoint"):
        encoder_class.from_pretrained(None)


@pytest.mark.parametrize("class_name", ["PrimusM", "Primus"])
def test_preset_and_plain_primus_plans_are_built(class_name):
    arch_kwargs = None
    if class_name == "Primus":
        arch_kwargs = {
            "embed_dim": 24,
            "patch_embed_size": [4, 4, 4],
            "eva_depth": 1,
            "eva_numheads": 2,
        }
    network = primus.build_primus(
        {"arch_class_name": class_name, "arch_kwargs": arch_kwargs}, 1, (16, 16, 16)
    )
    assert network.embed_dim == (864 if class_name == "PrimusM" else 24)


def test_resenc_encoder_keeps_the_last_encoder_stage(tmp_path):
    path, weights = tiny_resenc_checkpoint(tmp_path)
    network = resenc.build_resenc(TINY_RESENC_PLANS, 1)
    network.load_state_dict(weights)

    encoder = ResEncEncoder.from_pretrained(path)

    assert not encoder.training
    assert encoder.embed_dim == 16 and encoder.downsample_factor == (4, 4, 4)
    assert encoder.patch_size == (16, 16, 16)
    assert encoder.target_spacing is None and encoder.normalization == "z_score"
    assert set(encoder.state_dict()) == {k for k in weights if k.startswith("encoder.")}
    x = torch.randn(2, 1, 16, 16, 16)
    with torch.no_grad():
        grid = encoder(x)
        torch.testing.assert_close(grid, network.eval().encoder(x)[-1])
    assert grid.shape == (2, 16, 4, 4, 4)


def test_resenc_l_preset_is_built():
    network = resenc.build_resenc({"arch_class_name": "ResEncL", "arch_kwargs": None}, 1)
    assert network.encoder.output_channels[-1] == 320


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


def make_dataset(input_dir, mask_dir=None, orientation="LPS"):
    preprocessing = {
        "target_spacing": (1.0, 1.0, 1.0),
        "normalization": "z_score",
        "input_shape": (8, 8, 8),
        "processing_mode": "resized",
    }
    files = nifti_inputs.find_nifti_files(input_dir)
    return nifti_inputs.NiftiEncoderInputDataset(
        input_dir, files, orientation, mask_dir, preprocessing
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


def test_nested_volumes_use_their_masks_and_failures_are_reported(tmp_path):
    identity = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    array = np.zeros((16, 16, 16), np.float32)
    array[4:8, 4:8, 4:8] = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    mask = np.zeros((16, 16, 16), np.uint8)
    mask[4:8, 4:8, 4:8] = 1
    write_volume(tmp_path / "in/a/b/scan.nii.gz", array, identity)
    write_volume(tmp_path / "masks/a/b/scan.nii.gz", mask, identity)
    (tmp_path / "in/broken.nii").write_text("not a volume")

    dataset = make_dataset(tmp_path / "in", mask_dir=tmp_path / "masks")
    scan, broken = dataset[0], dataset[1]

    assert broken["image_id"] == "broken" and broken["encoder_input"] is None
    assert broken["error"]
    assert scan["image_id"] == "a/b/scan" and scan["error"] is None
    # The 4³ mask ROI is enlarged to the 8³ input shape around it and normalized again.
    roi = torch.from_numpy(array[2:10, 2:10, 2:10])
    expected = ((roi - roi.mean()) / roi.std(correction=0))[None]
    torch.testing.assert_close(scan["encoder_input"], expected)
