"""Every paper recipe builds into a valid run config, as 'train.py' builds it."""

import csv
import re
from pathlib import Path

import pytest

from vlm.configs.run_config import RunConfig
from vlm.utils.serialization import load_config_file

PAPER_CONFIGS = Path(__file__).resolve().parents[2] / "paper_configs"


def launchable_configs():
    """(paper, train config, '--add_base_configs') of every run the docs describe."""
    rmr = PAPER_CONFIGS / "RMR"
    with open(rmr / "runs.csv") as runs:
        for row in csv.DictReader(runs):
            yield "RMR", row["train_config"], [row["add_base_configs"]]
    for encoder in ("nnFoundationViT", "nnFoundationCNN"):
        for dataset in ("ctrate", "merlin"):
            yield "nnFoundation", f"{encoder}.yml", [f"bases/datasets/{dataset}/{encoder}.yml"]


@pytest.fixture(autouse=True)
def config_variables(monkeypatch, tmp_path):
    """The '${NAME}' variables the recipes reference, which validation expands."""
    names = set()
    for path in PAPER_CONFIGS.rglob("*.yml"):
        names.update(re.findall(r"\$\{([A-Z_]+)\}", path.read_text()))
    for name in names:
        monkeypatch.setenv(name, str(tmp_path / name))


# Deprecated fields only warn when loaded; the recipes should have none.
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("paper, train_config, add_base_configs", list(launchable_configs()))
def test_paper_config_builds(paper, train_config, add_base_configs):
    config_dir = PAPER_CONFIGS / paper
    cfg_dict = load_config_file(config_dir / train_config)
    cfg_dict["base_configs"] = add_base_configs + (cfg_dict.get("base_configs") or [])
    cfg_dict = RunConfig.apply_base_configs(cfg_dict=cfg_dict, config_dir=config_dir)
    cfg_dict.setdefault("general", {})["train_config"] = str(config_dir / train_config)

    cfg = RunConfig.validate_with_cli_args(cfg_dict, [])
    cfg.apply_conditional_resets()
