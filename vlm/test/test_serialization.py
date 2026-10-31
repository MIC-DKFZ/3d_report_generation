"""Saved YAML files are replaced atomically, so concurrent readers never see a partial file."""

import yaml

from vlm.utils import serialization
from vlm.utils.serialization import load_yaml, save_yaml


def test_yaml_is_replaced_only_after_it_is_fully_written(tmp_path, monkeypatch):
    path = tmp_path / "final_config.yml"
    save_yaml({"run": {"step": 1}}, path)
    contents_while_writing = []
    safe_dump = yaml.safe_dump

    def dump_and_look(obj, stream, **kwargs):
        # A reader in another process, e.g. a rank loading the config of a resumed run.
        contents_while_writing.append(load_yaml(path))
        return safe_dump(obj, stream, **kwargs)

    monkeypatch.setattr(serialization.yaml, "safe_dump", dump_and_look)
    save_yaml({"run": {"step": 2}}, path)

    assert contents_while_writing == [{"run": {"step": 1}}]
    assert load_yaml(path) == {"run": {"step": 2}}
    assert [p.name for p in tmp_path.iterdir()] == ["final_config.yml"]
