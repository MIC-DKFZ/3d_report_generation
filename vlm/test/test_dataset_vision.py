"""Check visual shape preparation after separating it from dataset sampling."""

from types import SimpleNamespace

import numpy as np
import pytest

from vlm.data.datasets import MedicalImageDataset


def load_embedding(path, *, stacked=False, slices=None):
    dataset = SimpleNamespace(
        vision_input_type="embedding",
        do_resize=False,
        do_transforms=False,
        vision_encoder_patch_size=None,
        split="test",
        training_transforms=None,
        is_stacked_2d_embedding=stacked,
        reduce_to_n_slices=slices,
    )
    return MedicalImageDataset._load_vision_input(dataset, path)


@pytest.mark.parametrize("suffix", ["npy", "npz"])
def test_volume_embedding_gets_channel_dimension(tmp_path, suffix):
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / f"embedding.{suffix}"
    if suffix == "npy":
        np.save(path, array)
    else:
        np.savez(path, features=array)

    actual = load_embedding(path)

    np.testing.assert_array_equal(actual, array[None])
    assert actual.dtype == np.float32


def test_channelled_embedding_is_preserved(tmp_path):
    array = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
    path = tmp_path / "embedding.npy"
    np.save(path, array)

    np.testing.assert_array_equal(load_embedding(path), array)


def test_stacked_embeddings_drop_class_tokens_and_pool_slices(tmp_path):
    # Four slices, one class token plus four spatial tokens, two feature channels.
    array = np.zeros((4, 5, 2), dtype=np.float32)
    array[:, 0, :] = -999
    for slice_index in range(4):
        array[slice_index, 1:, 0] = slice_index
        array[slice_index, 1:, 1] = slice_index + 10
    path = tmp_path / "embedding.npy"
    np.save(path, array)

    actual = load_embedding(path, stacked=True, slices=2)

    expected = np.empty((2, 2, 2, 2), dtype=np.float32)
    expected[0, 0] = 0.5
    expected[0, 1] = 2.5
    expected[1, 0] = 10.5
    expected[1, 1] = 12.5
    np.testing.assert_array_equal(actual, expected)
