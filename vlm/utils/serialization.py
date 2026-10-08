import json
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
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, indent=indent, **kwargs)
