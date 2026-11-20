"""Embedding presets, multi-block inputs and the embedding runner (vlm/embeddings)."""

import blosc2
import numpy as np
import SimpleITK as sitk
import torch
from test_nifti_inference import tiny_checkpoint, write_volume

from vlm.data.vision_input_settings import load_settings
from vlm.embeddings.multiblock import (
    merge_block_embeddings,
    preprocess_multiblock_input,
)
from vlm.embeddings.runner import run
from vlm.models.vision_encoders.primus import PrimusEncoder
from vlm.scripts.embeddings.generate_embeddings import embedding_job, parse_args

RAS = (-1, 0, 0, 0, -1, 0, 0, 0, 1)


def test_multiblock_embeddings_are_merged_without_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint, _ = tiny_checkpoint(tmp_path)
    rng = np.random.default_rng(0)
    mask = np.zeros((40, 22, 36), np.uint8)
    mask[3:36, 4:18, 2:30] = 10
    for name in ("train_fixed/x_1", "valid_fixed/y_1"):
        volume = rng.normal(size=mask.shape).astype(np.float32)
        write_volume(tmp_path / f"in/{name}.nii.gz", volume, RAS)
        write_volume(tmp_path / f"masks/{name}.nii.gz", mask, RAS)

    job = embedding_job(
        parse_args(
            [
                *("--embedding", "Primus-4B_2_1_2_ROI", "--dataset", "ctrate"),
                *("--input_dir", str(tmp_path / "in"), "--mask_dir", str(tmp_path / "masks")),
                *("--output_dir", str(tmp_path / "out"), "--checkpoint", str(checkpoint)),
                *("--num_workers", "0", "--save_workers", "0", "--block_batch_size", "3"),
                "--debug_save_preprocessed",
            ]
        )
    )
    run(job)

    assert load_settings(tmp_path / "out") is None
    debug = sorted(p.name for p in (tmp_path / "out/valid_fixed/preprocessed").iterdir())
    assert debug == [f"y_1_block{i}.nii.gz" for i in range(4)]
    encoder = PrimusEncoder.from_pretrained(checkpoint)
    blocks, offsets = preprocess_multiblock_input(
        sitk.DICOMOrient(sitk.ReadImage(tmp_path / "in/valid_fixed/y_1.nii.gz"), "LPS"),
        sitk.DICOMOrient(sitk.ReadImage(tmp_path / "masks/valid_fixed/y_1.nii.gz"), "LPS"),
        **job.settings.vision_input_preprocessing.preprocessing_kwargs(encoder.patch_size),
        grid=(2, 1, 2),
    )
    with torch.no_grad():
        expected = merge_block_embeddings(encoder(blocks), offsets, encoder.patch_size, (2, 1, 2))
    saved = blosc2.load_array(str(tmp_path / "out/valid_fixed/y_1_embedding.bl2"))
    assert saved.shape == (24, 8, 4, 8)
    torch.testing.assert_close(torch.from_numpy(saved), expected)
