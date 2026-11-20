"""Portable ConfigPath declarations, runtime expansion, and saved expressions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vlm.configs.paths import ConfigPath
from vlm.configs.run_config import RunConfig
from vlm.training.checkpointing import load_run_config_from_run_dir, save_final_config
from vlm.utils.serialization import load_yaml, save_yaml


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
                "class_name": "AdapterDWUScheduler",
                "class_kwargs": {"adapter_lr": 0.0001, "lm_lr": 0.0003},
                "optimizer": {"class_name": "AdamW", "class_kwargs": {"lr": 0.001}},
            },
            "eval": {"run_inference": False, "run_eval": False},
        }
    )


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


def test_training_setup_preserves_train_config_handling_and_portable_output(
    monkeypatch, cfg, tmp_path
):
    import train

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("VLM_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("VLM_OUTPUT_DIR", str(tmp_path / "results"))
    raw = cfg.model_dump()
    raw["general"]["output_dir"] = None
    raw["general"]["run_name"] = None
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
            "--general.run_name_suffix",
            "nv_reason_ct",
        ],
    )
    log_files = []
    monkeypatch.setattr(train, "setup_logger", log_files.append)
    monkeypatch.setenv("SLURM_JOB_ID", "1234567")

    loaded, _ = train.setup_config_and_logging(False, 1, 1)

    # The CLI suffix names the run directory, which holds the log and the input config.
    assert loaded.general.run_name.startswith("run__nv_reason_ct__")
    assert loaded.general.run_name.endswith("__1234567")
    run_dir = Path(loaded.general.output_dir.get_resolved_path())
    assert run_dir.name == loaded.general.run_name
    assert log_files == [str(run_dir / train.TRAIN_LOG_FILENAME)]
    assert (run_dir / train.INPUT_CONFIG_FILENAME).exists()
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


def test_inference_and_evaluation_need_neither_root_variable(tmp_path):
    # Both roots are read only where needed: inference finds the run by its path, and
    # evaluation reads saved predictions.
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("VLM_CONFIG_DIR", "VLM_OUTPUT_DIR")
    }
    code = (
        "import eval, inference\n"
        "from vlm.utils.paths import resolve_run_dir\n"
        f"assert resolve_run_dir({str(tmp_path)!r}) == __import__('pathlib').Path({str(tmp_path)!r})\n"
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
