"""Focused tests for CLI overrides of dynamic class keyword arguments."""

import pytest

from vlm.configs.data_config import DataConfig
import vlm.configs.run_config as run_config
from vlm.configs.run_config import RunConfig
from vlm.data import datasets
from vlm.training.checkpointing import load_run_config_from_run_dir, save_final_config
from vlm.utils.serialization import load_yaml, save_yaml


def make_run_config() -> RunConfig:
    return RunConfig.model_validate(make_run_config_dict())


def make_run_config_dict() -> dict:
    return {
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
            "class_name": "AdapterDWUScheduler",
            "class_kwargs": {"adapter_lr": 0.0001, "lm_lr": 0.0003},
            "optimizer": {
                "class_name": "AdamW",
                "class_kwargs": {"lr": 0.001, "weight_decay": 0.01, "amsgrad": True},
            },
        },
        "eval": {"run_inference": False, "run_eval": False},
    }


def test_dynamic_class_kwargs_overrides_preserve_types_and_siblings():
    cfg = make_run_config()
    cfg.merge_subconfigs_with_cli_args(
        [
            "--scheduler.class_kwargs.adapter_lr",
            "0.0002",
            "--scheduler.optimizer.class_kwargs.no-amsgrad",
            "--scheduler.optimizer.class_kwargs.lr",
            "0.005",
        ]
    )

    assert cfg.scheduler.class_kwargs == {"adapter_lr": 0.0002, "lm_lr": 0.0003}
    assert cfg.scheduler.optimizer.class_kwargs == {
        "lr": 0.005,
        "weight_decay": 0.01,
        "amsgrad": False,
    }
    assert isinstance(cfg.scheduler.optimizer.class_kwargs["lr"], float)


def test_new_dynamic_class_kwargs_can_be_added():
    cfg = make_run_config()
    del cfg.scheduler.optimizer.class_kwargs["amsgrad"]
    del cfg.scheduler.optimizer.class_kwargs["lr"]

    cfg.merge_subconfigs_with_cli_args(
        [
            "--scheduler.optimizer.class_kwargs.amsgrad",
            "false",
            "--scheduler.optimizer.class_kwargs.lr",
            "0.005",
            "--scheduler.optimizer.class_kwargs.betas",
            "[0.9, 0.999]",
        ]
    )

    assert cfg.scheduler.optimizer.class_kwargs["amsgrad"] is False
    assert cfg.scheduler.optimizer.class_kwargs["lr"] == 0.005
    assert isinstance(cfg.scheduler.optimizer.class_kwargs["lr"], float)
    assert cfg.scheduler.optimizer.class_kwargs["betas"] == (0.9, 0.999)
    assert cfg.scheduler.optimizer.class_kwargs["weight_decay"] == 0.01


def test_plain_dict_field_keys_can_be_overridden_and_added():
    cfg = make_run_config()

    cfg.merge_subconfigs_with_cli_args(
        ["--trainer.gradient_checkpointing_kwargs.every_n_layers", "3"]
    )

    assert cfg.trainer.gradient_checkpointing_kwargs == {"use_reentrant": False, "every_n_layers": 3}


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


def make_unnamed_run_config_dict(monkeypatch, tmp_path) -> dict:
    # The run directory contains the train config's folder relative to 'VLM_CONFIG_DIR'.
    monkeypatch.setattr(run_config, "VLM_CONFIG_DIR", str(tmp_path / "configs"))
    monkeypatch.setenv("VLM_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "1234567")
    cfg_dict = make_run_config_dict()
    cfg_dict["general"]["train_config"] = str(tmp_path / "configs" / "config.yml")
    del cfg_dict["general"]["run_name"], cfg_dict["general"]["output_dir"]
    return cfg_dict


def test_cli_overrides_name_the_run_directory(monkeypatch, tmp_path):
    cfg = RunConfig.validate_with_cli_args(
        make_unnamed_run_config_dict(monkeypatch, tmp_path),
        ["--general.run_name_suffix", "nv_reason_ct", "--data.vision_input_path", "/tmp/other"],
    )

    run_name = cfg.general.run_name
    assert run_name.startswith("config__nv_reason_ct__") and run_name.endswith("__1234567")
    assert cfg.general.output_dir.get_resolved_path() == str(tmp_path / "ctrate" / "other" / run_name)


def test_run_name_suffix_does_not_change_an_explicit_run_name(monkeypatch, tmp_path):
    cfg_dict = make_unnamed_run_config_dict(monkeypatch, tmp_path)

    cfg = RunConfig.validate_with_cli_args(
        cfg_dict, ["--general.run_name", "explicit", "--general.run_name_suffix", "nv_reason_ct"]
    )

    assert cfg.general.run_name == "explicit"
    assert cfg.general.output_dir.get_resolved_path().endswith("/explicit")


def test_cli_overrides_keep_defaults_of_dict_fields_missing_from_the_config(
    monkeypatch, tmp_path
):
    cfg = RunConfig.validate_with_cli_args(
        make_unnamed_run_config_dict(monkeypatch, tmp_path),
        [
            "--trainer.gradient_checkpointing_kwargs.every_n_layers",
            "3",
            "--scheduler.class_kwargs.adapter_lr",
            "0.0002",
        ],
    )

    assert cfg.trainer.gradient_checkpointing_kwargs == {"use_reentrant": False, "every_n_layers": 3}
    assert cfg.scheduler.class_kwargs == {"adapter_lr": 0.0002, "lm_lr": 0.0003}


def test_validation_without_cli_overrides_matches_plain_validation(monkeypatch, tmp_path):
    cfg_dict = make_unnamed_run_config_dict(monkeypatch, tmp_path)

    cfg = RunConfig.validate_with_cli_args(cfg_dict, [])

    assert cfg.general.run_name.startswith("config__") and "nv_reason_ct" not in cfg.general.run_name
    assert cfg.model_dump(exclude={"general"}) == RunConfig.model_validate(cfg_dict).model_dump(
        exclude={"general"}
    )


def test_validation_rejects_unknown_cli_overrides(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="2"):
        RunConfig.validate_with_cli_args(
            make_unnamed_run_config_dict(monkeypatch, tmp_path), ["--general.seeed", "3"]
        )
