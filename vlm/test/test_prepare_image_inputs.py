"""Preparation of image-mode encoder inputs from NIfTI volumes."""

import numpy as np
import SimpleITK as sitk
import yaml

from vlm.configs.data_config import DataConfig
from vlm.data.datasets.vision import load_vision_input
from vlm.data.preprocessing import prepare_image_inputs
from vlm.data.preprocessing.encoder_input import preprocess_encoder_input
from vlm.data.types import VisionInputType


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
