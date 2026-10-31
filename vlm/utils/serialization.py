import json
import os
import yaml
from pathlib import Path
from typing import Any


def load_config_file(config_path: str | Path) -> dict:
    file_ext = Path(config_path).suffix.lower()
    if file_ext in (".yaml", ".yml"):
        return load_yaml(config_path)
    if file_ext == ".json":
        return load_json(config_path)
    raise ValueError(f"Unsupported config file format: {file_ext}")


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 4, **kwargs) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, **kwargs)


def load_yaml(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Any, path: str | Path, indent: int = 2, **kwargs) -> None:
    """
    Write atomically: other processes read either the old or the new file, never a partial one,
    e.g. ranks loading 'final_config.yml' of a resumed run while rank 0 rewrites it.
    """
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, indent=indent, **kwargs)
    os.replace(temporary_path, path)
