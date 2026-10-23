"""Focused tests for CLI overrides of dynamic class keyword arguments."""

import pytest

from vlm.configs.data_config import DataConfig
from vlm.configs.run_config import RunConfig
from vlm.data import datasets
from vlm.training.checkpointing import load_run_config_from_run_dir, save_final_config
from vlm.utils.serialization import load_yaml, save_yaml


def make_run_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "general": {
                "train_config": "/tmp/config.yml",
                "run_name": "kwargs-check",
                "output_dir": "/tmp/kwargs-check",
            },
            "model": {
                "model_type": "stub",
                "model_id": "stub",
                "trainable_params": {
                    "language_model": {"mode": "full"},
                    "lm_head": {"mode": "full"},
                },
            },
            "data": {
                "dataset_name": "ctrate",
                "vision_input_path": "/tmp/input",
                "answer_types": "all",
                "text_only": True,
            },
            "trainer": {},
            "scheduler": {
                "class_name": "AdapterDWUConstScheduler",
                "class_kwargs": {
                    "adapter_lr": 0.0001,
                    "lm_lr": 0.0003,
                    "reset_optimizer": True,
                },
                "optimizer": {
                    "class_name": "AdamW",
                    "class_kwargs": {"lr": 0.001, "weight_decay": 0.01},
                },
            },
            "eval": {"run_inference": False, "run_eval": False},
        }
    )


def test_dynamic_class_kwargs_overrides_preserve_types_and_siblings():
    cfg = make_run_config()
    cfg.merge_subconfigs_with_cli_args(
        [
            "--scheduler.class_kwargs.adapter_lr",
            "0.0002",
            "--scheduler.class_kwargs.no-reset_optimizer",
            "--scheduler.optimizer.class_kwargs.lr",
            "0.005",
        ]
    )

    assert cfg.scheduler.class_kwargs == {
        "adapter_lr": 0.0002,
        "lm_lr": 0.0003,
        "reset_optimizer": False,
    }
    assert cfg.scheduler.optimizer.class_kwargs == {"lr": 0.005, "weight_decay": 0.01}
    assert isinstance(cfg.scheduler.optimizer.class_kwargs["lr"], float)


def test_new_dynamic_class_kwargs_can_be_added():
    cfg = make_run_config()
    del cfg.scheduler.class_kwargs["reset_optimizer"]
    del cfg.scheduler.optimizer.class_kwargs["lr"]

    cfg.merge_subconfigs_with_cli_args(
        [
            "--scheduler.class_kwargs.reset_optimizer",
            "false",
            "--scheduler.optimizer.class_kwargs.lr",
            "0.005",
            "--scheduler.optimizer.class_kwargs.betas",
            "[0.9, 0.999]",
        ]
    )

    assert cfg.scheduler.class_kwargs["reset_optimizer"] is False
    assert cfg.scheduler.optimizer.class_kwargs["lr"] == 0.005
    assert isinstance(cfg.scheduler.optimizer.class_kwargs["lr"], float)
    assert cfg.scheduler.optimizer.class_kwargs["betas"] == (0.9, 0.999)
    assert cfg.scheduler.optimizer.class_kwargs["weight_decay"] == 0.01


def test_unknown_non_class_kwarg_is_rejected():
    cfg = make_run_config()
    with pytest.raises(SystemExit, match="2"):
        cfg.merge_subconfigs_with_cli_args(["--general.seeed", "3"])


def test_automatic_dataset_selection_follows_cli_overrides():
    cfg = make_run_config()
    assert cfg.data.get_dataset_class() is datasets.CTRATEDataset

    cfg.merge_subconfigs_with_cli_args(["--data.retrieval_json_path", "/tmp/retrieval.json"])
    assert cfg.data.get_dataset_class() is datasets.CTRATEDatasetRetrieval
    assert "dataset_class_name" not in cfg.data.model_dump()

    cfg.merge_subconfigs_with_cli_args(
        ["--data.retrieval_use_image_in_prompt", "--data.no-text_only"]
    )
    assert cfg.data.get_dataset_class() is datasets.CTRATEDatasetRetrievalWithImage
    assert "dataset_class_name" not in cfg.data.model_dump()

    cfg.merge_subconfigs_with_cli_args(["--data.dataset_name", "merlin"])
    assert cfg.data.get_dataset_class() is datasets.MerlinDatasetRetrievalWithImage

    cfg.merge_subconfigs_with_cli_args(["--data.no-retrieval_use_image_in_prompt"])
    assert cfg.data.get_dataset_class() is datasets.MerlinDatasetRetrieval
    assert "dataset_class_name" not in cfg.data.model_dump()


def test_removing_retrieval_after_serialization_restores_plain_dataset():
    cfg = make_run_config()
    cfg.merge_subconfigs_with_cli_args(["--data.retrieval_json_path", "/tmp/retrieval.json"])
    assert cfg.data.get_dataset_class() is datasets.CTRATEDatasetRetrieval

    saved_data = cfg.data.model_dump()
    saved_data["retrieval_json_path"] = None
    reloaded = DataConfig.model_validate(saved_data)
    assert reloaded.get_dataset_class() is datasets.CTRATEDataset
    assert "dataset_class_name" not in reloaded.model_dump()


@pytest.mark.parametrize(
    "legacy_field,class_name", [(False, None), (True, None), (True, "CTRATEDataset")]
)
def test_saved_dataset_selection_follows_current_settings(tmp_path, legacy_field, class_name):
    cfg = make_run_config()
    cfg.data.text_only = False
    save_final_config(cfg, output_dir=tmp_path)
    saved = load_yaml(tmp_path / "final_config.yml")
    assert "dataset_class_name" not in saved["run"]["data"]
    if legacy_field:
        saved["run"]["data"]["dataset_class_name"] = class_name
        save_yaml(saved, tmp_path / "final_config.yml")

    reloaded, _ = load_run_config_from_run_dir(tmp_path)
    reloaded.merge_subconfigs_with_cli_args(
        ["--data.retrieval_json_path", "/tmp/retrieval.json", "--data.retrieval_use_image_in_prompt"]
    )
    assert reloaded.data.get_dataset_class() is datasets.CTRATEDatasetRetrievalWithImage
    assert "dataset_class_name" not in reloaded.data.model_dump()
    assert reloaded.general.run_name == cfg.general.run_name
    assert reloaded.general.output_dir.get_resolved_path() == str(tmp_path)
    save_final_config(reloaded)
    assert "dataset_class_name" not in load_yaml(tmp_path / "final_config.yml")["run"]["data"]


def test_cli_rejects_removed_dataset_class_override():
    cfg = make_run_config()
    with pytest.raises(SystemExit, match="2"):
        cfg.merge_subconfigs_with_cli_args(
            ["--data.dataset_class_name", "CTRATEDatasetRetrievalWithImagePrompt1"]
        )
