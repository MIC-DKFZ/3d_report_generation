"""The settings file written by the embedding script (vlm/scripts/embeddings)."""

import importlib
import sys
import types
from argparse import Namespace

import pytest

from vlm.data.vision_input_settings import load_settings


@pytest.fixture
def embedding_script(monkeypatch):
    # The encoders come from an external 'models' package, not needed for the settings.
    for name, attribute in [
        ("dynamic_arch", "Dynamic3DEncoder"),
        ("VoCo", "VoCoEncoder"),
        ("SwinUNetR", "SwinUNetREncoder"),
        ("MisFM", "MisFMEncoder"),
        ("vista3d", "Vista3DEncoder"),
        ("curia", "CuriaEncoder"),
        ("merlin", "MerlinEncoder"),
        ("CTFM", "CTFMEncoder"),
    ]:
        module = types.ModuleType(f"models.{name}")
        setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, f"models.{name}", module)
    monkeypatch.setitem(sys.modules, "models", types.ModuleType("models"))
    monkeypatch.delitem(
        sys.modules, "vlm.scripts.embeddings.embedd_images_baselines", raising=False
    )
    return importlib.import_module("vlm.scripts.embeddings.embedd_images_baselines")


def script_args(**overrides):
    args = {
        "orientation": "lps",
        "grid": [1, 1, 1],
        "processing_mode": "resized",
        "no_mask": False,
        "mask_folder": "/data/masks",
        "mask_label_ids": [],
        "enforce_min_bbox": True,
        "outlier_method": "connected_components",
        "outlier_min_voxels": 100,
    }
    return Namespace(**{**args, **overrides})


def test_single_block_embeddings_get_their_settings(embedding_script, tmp_path):
    config = embedding_script.get_model_config("small_resencL")

    settings, reason = embedding_script.vision_input_settings(script_args(), config)

    assert reason is None and settings.vision_input_orientation == "LPS"
    preprocessing = settings.vision_input_preprocessing
    assert preprocessing.target_spacing == (1.0, 1.0, 1.0)
    assert preprocessing.processing_mode == "resized"
    assert (
        preprocessing.roi_mask == "masks in /data/masks" and preprocessing.mask_label_ids is None
    )
    assert preprocessing.outlier_method == "connected_components"
    embedding_script.save_settings(tmp_path, settings)
    assert load_settings(tmp_path) == settings


def test_whole_volume_embeddings_have_no_mask_settings(embedding_script):
    config = embedding_script.get_model_config("merlin")

    settings, _ = embedding_script.vision_input_settings(
        script_args(no_mask=True, mask_folder=None, processing_mode="center_192crop"), config
    )

    assert settings is None  # center_192crop of a 160x224x224 encoder
    settings, _ = embedding_script.vision_input_settings(
        script_args(no_mask=True, mask_folder=None), config
    )
    assert settings.vision_input_preprocessing.roi_mask is None
    assert settings.vision_input_preprocessing.outlier_method == "none"
    assert settings.vision_input_preprocessing.normalization == "merlin_scale"


@pytest.mark.parametrize(
    "overrides, reason",
    [({"orientation": None}, "--orientation"), ({"grid": [2, 2, 2]}, "multi-block")],
)
def test_embeddings_without_settings_say_why(embedding_script, overrides, reason):
    config = embedding_script.get_model_config("big_primus")

    settings, why = embedding_script.vision_input_settings(script_args(**overrides), config)

    assert settings is None and reason in why
