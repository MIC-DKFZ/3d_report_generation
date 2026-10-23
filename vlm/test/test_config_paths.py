"""Portable ConfigPath declarations, runtime expansion, and saved expressions."""

import os
import sys

import pytest
from pydantic import Field, ValidationError

from vlm.configs.base import ConfigBase
from vlm.configs.paths import ConfigPath
from vlm.configs.run_config import RunConfig
from vlm.training.checkpointing import load_run_config_from_run_dir, save_final_config
from vlm.utils.serialization import load_yaml, save_yaml


class PathConfig(ConfigBase):
    required: ConfigPath = Field(description="A required path.")
    optional: ConfigPath | None = None
    items: list[ConfigPath] | None = None
    prompt: str = "${LEAVE_THIS_ALONE}"


class NestedConfig(ConfigBase):
    paths: PathConfig


@pytest.fixture
def cfg(tmp_path):
    return RunConfig.model_validate(
        {
            "general": {
                "train_config": str(tmp_path / "run.yml"),
                "run_name": "portable-paths",
                "output_dir": str(tmp_path),
            },
            "model": {
                "model_type": "stub",
                "model_id": "organization/model",
                "trainable_params": {
                    "language_model": {"mode": "full"},
                    "lm_head": {"mode": "full"},
                },
            },
            "data": {
                "dataset_name": "ctrate",
                "answer_types": "all",
                "text_only": True,
                "train_dataset_json_path": "${DATA_ROOT}/train.json",
                "vision_input_path": "${DATA_ROOT}/embeddings",
            },
            "trainer": {},
            "scheduler": {
                "class_name": "AdapterDWUConstScheduler",
                "class_kwargs": {"adapter_lr": 0.0001, "lm_lr": 0.0003},
                "optimizer": {"class_name": "AdamW", "class_kwargs": {"lr": 0.001}},
            },
            "eval": {"run_inference": False, "run_eval": False},
        }
    )


def test_nested_models_and_lists_save_expressions_without_environment(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    cfg = NestedConfig(
        paths=PathConfig(
            required="${DATA_ROOT}/train.json",
            optional="${DATA_ROOT}/val.json",
            items=["${DATA_ROOT}/base.yml", "relative.yml"],
        )
    )
    assert cfg.model_dump()["paths"] == {
        "required": "${DATA_ROOT}/train.json",
        "optional": "${DATA_ROOT}/val.json",
        "items": ["${DATA_ROOT}/base.yml", "relative.yml"],
        "prompt": "${LEAVE_THIS_ALONE}",
    }


def test_validation_and_serialization_preserve_strings_without_environment(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    cfg = PathConfig(required="${DATA_ROOT}/train.json")
    assert isinstance(cfg.required, ConfigPath)
    assert cfg.required.expression == "${DATA_ROOT}/train.json"
    assert cfg.model_dump()["required"] == "${DATA_ROOT}/train.json"
    assert PathConfig.model_validate_json(cfg.model_dump_json()).required == cfg.required
    with pytest.raises(ValueError, match=r"required.*DATA_ROOT"):
        cfg.required.get_resolved_path(field="required")


def test_pydantic_accepts_existing_objects_and_exposes_a_string_schema():
    path = ConfigPath("relative/train.json")
    cfg = PathConfig(required=path)
    assert cfg.required is path
    assert str(cfg.required) == "relative/train.json"
    assert PathConfig.model_json_schema()["properties"]["required"]["type"] == "string"
    with pytest.raises(ValidationError):
        PathConfig(required=123)


def test_runtime_override_does_not_survive_serialization_or_mutate_deep_copy(monkeypatch):
    monkeypatch.delenv("UNSET_ROOT", raising=False)
    cfg = PathConfig(required="${UNSET_ROOT}/train.json")
    copy = cfg.model_copy(deep=True)
    copy.required.set_runtime_path("/scratch/train.json")
    assert copy.required.get_resolved_path() == "/scratch/train.json"
    assert cfg.required.expression == "${UNSET_ROOT}/train.json"
    with pytest.raises(ValueError, match="UNSET_ROOT"):
        cfg.required.get_resolved_path()
    loaded = PathConfig.model_validate_json(copy.model_dump_json())
    assert loaded.required.expression == "${UNSET_ROOT}/train.json"
    with pytest.raises(ValueError, match="UNSET_ROOT"):
        loaded.required.get_resolved_path()


@pytest.mark.parametrize("value", [None, "None", " null "])
def test_disabled_validation_dataset_remains_none(cfg, value):
    raw = cfg.model_dump()
    raw["data"]["val_dataset_json_path"] = value
    loaded = RunConfig.model_validate(raw)
    assert loaded.data.val_dataset_json_path is None


def test_default_model_id_is_a_config_path(cfg):
    raw = cfg.model_dump()
    raw["model"]["model_type"] = "Qwen3-1.7B"
    raw["model"]["model_id"] = None
    loaded = RunConfig.model_validate(raw)
    assert isinstance(loaded.model.model_id, ConfigPath)
    assert loaded.model.model_id.get_resolved_path() == "Qwen/Qwen3-1.7B"
    assert loaded.model_dump()["model"]["model_id"] == "Qwen/Qwen3-1.7B"


def test_round_trip_ignores_old_snapshot_and_uses_new_cluster(monkeypatch, cfg, tmp_path):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    save_final_config(cfg)
    saved = load_yaml(tmp_path / "final_config.yml")
    assert saved["run"]["data"]["train_dataset_json_path"] == "${DATA_ROOT}/train.json"
    assert "resolved_paths" not in saved
    saved["resolved_paths"] = {"data.train_dataset_json_path": "/cluster_a/data/train.json"}
    save_yaml(saved, tmp_path / "final_config.yml")
    monkeypatch.setenv("DATA_ROOT", "/cluster_b/data")
    loaded, _ = load_run_config_from_run_dir(tmp_path)
    assert loaded.data.train_dataset_json_path.get_resolved_path() == "/cluster_b/data/train.json"


def test_saving_after_staging_preserves_shared_source_paths(monkeypatch, cfg, tmp_path):
    monkeypatch.setenv("DATA_ROOT", "/shared/data")
    cfg.data.vision_input_path.set_runtime_path(tmp_path / "scratch")
    assert cfg.data.vision_input_path.get_resolved_path() == str(tmp_path / "scratch")
    save_final_config(cfg)
    saved = load_yaml(tmp_path / "final_config.yml")
    assert saved["run"]["data"]["vision_input_path"] == "${DATA_ROOT}/embeddings"
    assert "resolved_paths" not in saved
    assert cfg.data.vision_input_path.get_resolved_path() == str(tmp_path / "scratch")
    loaded = RunConfig.model_validate(saved["run"])
    assert loaded.data.vision_input_path.get_resolved_path() == "/shared/data/embeddings"


def test_cli_and_assignment_keep_expressions(monkeypatch, cfg):
    monkeypatch.setenv("DATA_ROOT", "/data")
    cfg.merge_subconfigs_with_cli_args(
        ["--data.train_dataset_json_path", "${DATA_ROOT}/override.json"]
    )
    assert cfg.data.train_dataset_json_path.expression == "${DATA_ROOT}/override.json"
    assert cfg.data.train_dataset_json_path.get_resolved_path() == "/data/override.json"
    cfg.data.train_dataset_json_path = ConfigPath("${DATA_ROOT}/assigned.json")
    assert cfg.data.train_dataset_json_path.get_resolved_path() == "/data/assigned.json"


@pytest.mark.parametrize("value", ["${DATA_ROOT:-/fallback}/file", "${BROKEN"])
def test_invalid_expression_has_field_context(value):
    with pytest.raises(ValueError, match="dataset.*invalid path expression"):
        ConfigPath(value).get_resolved_path(field="dataset")


@pytest.mark.parametrize("value", [None, ""])
def test_missing_or_empty_variable_has_field_context(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DATA_ROOT", raising=False)
    else:
        monkeypatch.setenv("DATA_ROOT", value)
    with pytest.raises(ValueError, match="dataset.*DATA_ROOT.*unset or empty"):
        ConfigPath("${DATA_ROOT}/file").get_resolved_path(field="dataset")


def test_expansion_is_single_pass_and_preserves_identifiers(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/data/${LITERAL}")
    assert (
        ConfigPath("${DATA_ROOT}/file").get_resolved_path(field="path") == "/data/${LITERAL}/file"
    )
    assert (
        ConfigPath("organization/model").get_resolved_path(field="model_id")
        == "organization/model"
    )
    assert ConfigPath("previous-run").get_resolved_path(field="checkpoint") == "previous-run"
    assert ConfigPath("relative/file").get_resolved_path(field="path") == "relative/file"
    assert ConfigPath("~/file").get_resolved_path(field="path") == os.path.expanduser("~/file")


def test_base_configs_remain_plain_strings_and_preserve_dataset_expression(
    monkeypatch, tmp_path, cfg
):
    monkeypatch.setenv("BASE_ROOT", str(tmp_path))
    (tmp_path / "base.yml").write_text(
        "data:\n  train_dataset_json_path: '${DATA_ROOT}/train.json'\n"
    )
    with pytest.raises(FileNotFoundError):
        RunConfig.apply_base_configs({"base_configs": ["${BASE_ROOT}/base.yml"]}, tmp_path)
    merged = RunConfig.apply_base_configs({"base_configs": ["base.yml"]}, tmp_path)
    assert merged["data"]["train_dataset_json_path"] == "${DATA_ROOT}/train.json"
    raw = cfg.model_dump()
    raw["base_configs"] = merged["base_configs"]
    loaded = RunConfig.model_validate(raw)
    assert loaded.base_configs == ["base.yml"]
    assert loaded.model_dump()["base_configs"] == ["base.yml"]
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.delenv("BASE_ROOT")
    save_final_config(loaded)
    saved = load_yaml(tmp_path / "final_config.yml")
    assert saved["run"]["base_configs"] == ["base.yml"]


def test_dataset_reads_expanded_path_without_mutating_config(monkeypatch, cfg, tmp_path):
    from vlm.data.datasets import CTRATEDataset

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    (tmp_path / "train.json").write_text("[]")
    dataset = CTRATEDataset(cfg=cfg, split="train", vision_encoder_patch_size=None)
    assert dataset.dataset_json_path == tmp_path / "train.json"
    assert cfg.data.train_dataset_json_path.expression == "${DATA_ROOT}/train.json"


def test_moved_run_preserves_output_expression_and_legacy_configs_load(monkeypatch, cfg, tmp_path):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.delenv("MISSING_OLD_OUTPUT", raising=False)
    cfg.general.output_dir = ConfigPath("${MISSING_OLD_OUTPUT}/old_run")
    save_yaml({"run": cfg.model_dump()}, tmp_path / "final_config.yml")
    loaded, _ = load_run_config_from_run_dir(tmp_path)
    assert loaded.general.output_dir.expression == "${MISSING_OLD_OUTPUT}/old_run"
    assert loaded.general.output_dir.get_resolved_path() == str(tmp_path)
    save_final_config(loaded)
    saved = load_yaml(tmp_path / "final_config.yml")
    assert saved["run"]["general"]["output_dir"] == "${MISSING_OLD_OUTPUT}/old_run"
    assert "resolved_paths" not in saved
    assert loaded.general.output_dir.get_resolved_path() == str(tmp_path)


def test_eval_validation_does_not_access_filesystem(monkeypatch, cfg):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    raw = cfg.model_dump()
    raw["eval"].update(run_inference=True, run_eval=True)
    raw["data"]["test_dataset_json_path"] = "${DATA_ROOT}/test.json"
    assert (
        RunConfig.model_validate(raw).data.test_dataset_json_path.expression
        == "${DATA_ROOT}/test.json"
    )


def test_training_setup_preserves_train_config_handling_and_portable_output(
    monkeypatch, cfg, tmp_path
):
    import train
    import vlm.configs.run_config as run_config

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("VLM_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VLM_OUTPUT_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(train, "VLM_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(run_config, "VLM_CONFIG_DIR", str(tmp_path))
    raw = cfg.model_dump()
    raw["general"]["output_dir"] = None
    monkeypatch.delenv("UNUSED_TEST_ROOT", raising=False)
    raw["data"]["test_dataset_json_path"] = "${UNUSED_TEST_ROOT}/test.json"
    save_yaml(raw, tmp_path / "run.yml")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--train_config",
            "run.yml",
            "--data.train_dataset_json_path",
            "${DATA_ROOT}/override.json",
        ],
    )
    from loguru import logger

    handlers = []
    log_files = []

    def setup_logger(path):
        log_files.append(path)
        handlers.append(logger.add(path, format="{message}"))

    merge_overrides = RunConfig.merge_subconfigs_with_cli_args

    def merge_with_logging(self, cli_args):
        logger.info("Applying config overrides")
        return merge_overrides(self, cli_args)

    monkeypatch.setattr(train, "setup_logger", setup_logger)
    monkeypatch.setattr(RunConfig, "merge_subconfigs_with_cli_args", merge_with_logging)
    try:
        loaded, _ = train.setup_config_and_logging(False, 1)
    finally:
        for handler in handlers:
            logger.remove(handler)
    with open(log_files[0]) as log_file:
        assert "Applying config overrides" in log_file.read()
    saved = load_yaml(
        os.path.join(loaded.general.output_dir.get_resolved_path(), "final_config.yml")
    )
    assert loaded.general.train_config == str(tmp_path / "run.yml")
    assert saved["run"]["general"]["train_config"] == str(tmp_path / "run.yml")
    assert "resolved_paths" not in saved
    assert saved["run"]["general"]["output_dir"].startswith("${VLM_OUTPUT_DIR}/")
    assert saved["run"]["data"]["train_dataset_json_path"] == "${DATA_ROOT}/override.json"
    assert saved["run"]["data"]["test_dataset_json_path"] == "${UNUSED_TEST_ROOT}/test.json"
    assert loaded.general.output_dir.expression.startswith("${VLM_OUTPUT_DIR}/")
