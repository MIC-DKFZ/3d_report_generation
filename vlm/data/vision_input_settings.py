"""
The settings file of a dataset's vision inputs ('vision_input_preprocessing.yml' in
'vision_input_path'), written by the scripts that make the inputs and copied into the run config
by training, so that new volumes can be preprocessed like the training inputs.
"""

import os
from pathlib import Path

import yaml

from vlm.configs.data_config import VisionInputSettings

SETTINGS_FILE = "vision_input_preprocessing.yml"


def settings_path(vision_input_path: str | Path) -> Path:
    return Path(vision_input_path) / SETTINGS_FILE


def load_settings(vision_input_path: str | Path) -> VisionInputSettings | None:
    """The settings saved in 'vision_input_path', or None if it has none."""
    path = settings_path(vision_input_path)
    if not path.is_file():
        return None
    return VisionInputSettings.model_validate(yaml.safe_load(path.read_text()))


def save_settings(vision_input_path: str | Path, settings: VisionInputSettings) -> Path:
    """
    Save the settings in 'vision_input_path'. A dataset has one settings file for all its split
    folders, so settings that differ from saved ones are an error.
    """
    saved = load_settings(vision_input_path)
    path = settings_path(vision_input_path)
    if saved is not None and saved != settings:
        raise ValueError(
            f"{path} holds other settings than these:\n{to_yaml(saved)}All inputs of a dataset "
            "must be preprocessed the same way; use another folder for other settings."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomically, since parallel jobs (e.g. embedding shards) save the same settings.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(to_yaml(settings))
    os.replace(tmp_path, path)
    return path


def to_yaml(settings: VisionInputSettings) -> str:
    # JSON mode turns tuples into lists, which plain YAML can hold.
    return yaml.safe_dump(settings.model_dump(mode="json", exclude_none=False), sort_keys=False)
