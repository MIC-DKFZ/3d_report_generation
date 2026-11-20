"""The settings file of a dataset's vision inputs and its copy in the run config."""

import sys


from vlm.configs.data_config import DataConfig, VisionInputSettings
from vlm.data.vision_input_settings import save_settings
from vlm.utils.serialization import load_yaml, save_yaml

SETTINGS = VisionInputSettings(
    vision_input_orientation="LPS",
    vision_input_preprocessing={
        "processing_mode": "resized",
        "target_spacing": [1.0, 1.0, 1.0],
        "roi_mask": "TotalSegmentator lungs",
    },
)


def data_config(vision_input_path, **fields):
    return DataConfig(
        dataset_name="ctrate",
        answer_types="all",
        vision_input_path=str(vision_input_path),
        vision_input_type="embedding",
        embed_dim=8,
        embed_size=(2, 2, 2),
        **fields,
    )


def test_training_saves_the_settings_in_the_final_config(monkeypatch, tmp_path):
    import train

    dataset = tmp_path / "embeddings"
    save_settings(dataset, SETTINGS)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("VLM_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(train, "setup_logger", lambda path: None)
    save_yaml(
        {
            "general": {"run_name": "settings", "output_dir": str(tmp_path / "run")},
            "model": {
                "model_type": "stub",
                "model_id": "organization/model",
                "adapter": {"class_name": "QwenAdapter"},
                "trainable_params": {
                    "adapter": {"mode": "full"},
                    "language_model": {"mode": "full"},
                    "lm_head": {"mode": "full"},
                },
            },
            "data": {
                "dataset_name": "ctrate",
                "answer_types": "all",
                "train_dataset_json_path": "${DATA_ROOT}/train.json",
                "vision_input_path": "${DATA_ROOT}/embeddings",
                "vision_input_type": "embedding",
                "embed_dim": 8,
                "embed_size": [2, 2, 2],
            },
            "trainer": {},
            "scheduler": {
                "class_name": "AdapterDWUScheduler",
                "class_kwargs": {"adapter_lr": 0.0001, "lm_lr": 0.0003},
                "optimizer": {"class_name": "AdamW", "class_kwargs": {"lr": 0.001}},
            },
            "eval": {"run_inference": False, "run_eval": False},
        },
        tmp_path / "run.yml",
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--train_config", "run.yml"])

    loaded, _ = train.setup_config_and_logging(False, 1, 1)

    saved = load_yaml(tmp_path / "run/final_config.yml")["run"]["data"]
    assert saved["vision_input_orientation"] == "LPS"
    assert saved["vision_input_preprocessing"]["roi_mask"] == "TotalSegmentator lungs"
    # The path keeps its expression in the saved config.
    assert saved["vision_input_path"] == "${DATA_ROOT}/embeddings"
