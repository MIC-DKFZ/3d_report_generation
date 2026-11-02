"""Check visual shape preparation after separating it from dataset sampling."""

from types import SimpleNamespace

import numpy as np
import pytest

from vlm.data.datasets import MedicalImageDataset


def load_embedding(path, *, stacked=False, slices=None):
    dataset = SimpleNamespace(
        vision_input_type="embedding",
        vision_encoder_patch_size=None,
        training_transforms=None,
        is_stacked_2d_embedding=stacked,
        reduce_to_n_slices=slices,
    )
    return MedicalImageDataset._load_vision_input(dataset, path)


def load_image(path, patch_size=(4, 6, 8), transforms=None):
    dataset = SimpleNamespace(
        vision_input_type="image",
        vision_encoder_patch_size=patch_size,
        training_transforms=transforms,
        is_stacked_2d_embedding=False,
        reduce_to_n_slices=None,
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


@pytest.mark.parametrize("shape", [(4, 6, 8), (1, 4, 6, 8)])
def test_image_inputs_are_used_as_stored(tmp_path, shape):
    # Image-mode inputs are preprocessed before training: no normalization, resizing or cropping.
    array = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) * 100
    path = tmp_path / "image.npy"
    np.save(path, array)

    actual = load_image(path)

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, array.reshape(1, 4, 6, 8).astype(np.float32))


@pytest.mark.parametrize("shape", [(4, 6, 9), (2, 4, 6, 8), (8, 12, 16)])
def test_images_not_of_the_encoder_input_shape_are_rejected(tmp_path, shape):
    path = tmp_path / "image.npy"
    np.save(path, np.zeros(shape, dtype=np.float32))

    with pytest.raises(ValueError, match="prepare_image_inputs"):
        load_image(path)


def test_image_transforms_augment_the_input(tmp_path):
    path = tmp_path / "image.npy"
    np.save(path, np.ones((4, 6, 8), dtype=np.float32))
    seen = []

    def double(image):
        seen.append(tuple(image.shape))
        return {"image": image * 2}

    actual = load_image(path, transforms=double)

    assert seen == [(1, 4, 6, 8)]
    np.testing.assert_array_equal(actual, np.full((1, 4, 6, 8), 2, dtype=np.float32))


def test_nnunet_training_transforms_keep_the_input_shape(tmp_path):
    from vlm.data.image_transforms import get_training_image_transforms

    path = tmp_path / "image.npy"
    np.save(path, np.random.default_rng(0).normal(size=(16, 16, 16)).astype(np.float32))

    actual = load_image(path, (16, 16, 16), get_training_image_transforms((16, 16, 16)))

    assert actual.shape == (1, 16, 16, 16) and actual.dtype == np.float32
