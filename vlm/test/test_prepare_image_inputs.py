"""Preparation of image-mode encoder inputs from NIfTI volumes."""

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from vlm.configs.data_config import DataConfig, VisionInputSettings
from vlm.data.datasets.vision import load_vision_input
from vlm.data.preprocessing import prepare_image_inputs
from vlm.data.preprocessing.encoder_input import preprocess_encoder_input
from vlm.data.types import VisionInputType
from vlm.data.vision_input_settings import load_settings, save_settings


def write_volume(path, array, spacing=(1.0, 1.0, 1.0)):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def run(*argv):
    prepare_image_inputs.main(
        prepare_image_inputs.parse_args([*map(str, argv), "--num_workers", "0"])
    )


def test_volumes_are_saved_as_the_encoder_inputs_image_mode_loads(tmp_path):
    rng = np.random.default_rng(0)
    volumes = {name: rng.normal(size=(12, 10, 14)).astype(np.float32) for name in ("a", "b")}
    mask = np.zeros((12, 10, 14), np.uint8)
    mask[2:9, 3:8, 4:12] = 2
    for name, array in volumes.items():
        write_volume(tmp_path / f"in/{name}/{name}_1.nii.gz", array)
        write_volume(tmp_path / f"masks/{name}/{name}_1.nii.gz", mask)
    (tmp_path / "in/broken.nii").write_text("not a volume")

    run(
        "--input_dir", tmp_path / "in",
        "--vision_input_path", tmp_path / "out",
        "--split_folder", "train_fixed",
        "--mask_dir", tmp_path / "masks",
        "--roi_mask", "test organ",
        "--mask_label_ids", 2,
        "--orientation", "lps",
        "--input_shape", 8, 8, 8,
        "--processing_mode", "resized",
    )  # fmt: skip

    settings = yaml.safe_load((tmp_path / "out/vision_input_preprocessing.yml").read_text())
    assert not (tmp_path / "out/train_fixed/vision_input_preprocessing.yml").exists()
    data = DataConfig(dataset_name="ctrate", answer_types="all", **settings)
    assert data.vision_input_orientation == "LPS"
    assert data.vision_input_preprocessing.roi_mask == "test organ"
    assert data.vision_input_preprocessing.mask_label_ids == [2]
    kwargs = data.vision_input_preprocessing.preprocessing_kwargs((8, 8, 8))
    for name in volumes:
        loaded = load_vision_input(
            tmp_path / f"out/train_fixed/{name}_1.b2nd",
            vision_input_type=VisionInputType.IMAGE,
            is_stacked_2d_embedding=False,
            reduce_to_n_slices=None,
            patch_size=(8, 8, 8),
        )
        expected = preprocess_encoder_input(
            sitk.ReadImage(tmp_path / f"in/{name}/{name}_1.nii.gz"),
            sitk.ReadImage(tmp_path / f"masks/{name}/{name}_1.nii.gz"),
            **kwargs,
        )
        np.testing.assert_allclose(loaded, expected[0].numpy(), rtol=0, atol=1e-6)
    failed = yaml.safe_load((tmp_path / "out/train_fixed/failed.json").read_text())
    assert [entry["path"].endswith("broken.nii") for entry in failed] == [True]


def test_volumes_with_the_same_file_name_are_rejected(tmp_path):
    for folder in ("x", "y"):
        write_volume(tmp_path / f"in/{folder}/scan.nii.gz", np.zeros((4, 4, 4), np.float32))

    with pytest.raises(ValueError, match="share a file name"):
        run(
            "--input_dir", tmp_path / "in",
            "--vision_input_path", tmp_path / "out",
            "--orientation", "LPS",
            "--input_shape", 4, 4, 4,
            "--processing_mode", "center_crop",
        )  # fmt: skip


def test_masks_need_a_description():
    with pytest.raises(SystemExit):
        prepare_image_inputs.parse_args(
            ["--input_dir", "in", "--vision_input_path", "out", "--mask_dir", "masks"]
            + ["--orientation", "LPS", "--input_shape", "4", "4", "4"]
            + ["--processing_mode", "resized"]
        )


def test_the_split_folders_of_a_dataset_share_one_settings_file(tmp_path):
    for split in ("train", "valid"):
        write_volume(tmp_path / f"in_{split}/scan_{split}.nii.gz", np.ones((4, 4, 4), np.float32))
    args = ["--vision_input_path", tmp_path / "out", "--orientation", "LPS"]
    args += ["--input_shape", 4, 4, 4, "--processing_mode", "resized"]

    run("--input_dir", tmp_path / "in_train", "--split_folder", "train_fixed", *args)
    run("--input_dir", tmp_path / "in_valid", "--split_folder", "valid_fixed", *args)
    with pytest.raises(ValueError, match="other settings"):
        run(
            "--input_dir", tmp_path / "in_valid", "--split_folder", "valid_fixed", *args,
            "--target_spacing", 1, 1, 1, "--overwrite",
        )  # fmt: skip

    assert sorted(p.name for p in (tmp_path / "out").rglob("*.b2nd")) == [
        "scan_train.b2nd",
        "scan_valid.b2nd",
    ]
    assert load_settings(tmp_path / "out").vision_input_preprocessing.target_spacing is None


def test_settings_are_saved_and_loaded_unchanged(tmp_path):
    settings = VisionInputSettings(
        vision_input_orientation="las",
        vision_input_preprocessing={"processing_mode": "resized", "target_spacing": [2, 1, 1]},
    )

    save_settings(tmp_path, settings)

    assert load_settings(tmp_path) == settings
    assert load_settings(tmp_path).vision_input_orientation == "LAS"
    assert load_settings(tmp_path / "missing") is None
