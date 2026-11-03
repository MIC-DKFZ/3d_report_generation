import os
from pathlib import Path


def _required_env_var(name: str, purpose: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Set the environment variable {name}: it is the root {purpose}.")
    return value


def get_vlm_config_dir() -> str:
    """Root of the config paths given to training and to inference config overrides."""
    return _required_env_var("VLM_CONFIG_DIR", "that config paths are relative to")


def get_vlm_output_dir() -> str:
    """Root of new runs and of the search for runs given by their folder name."""
    return _required_env_var("VLM_OUTPUT_DIR", "for runs and for finding runs by folder name")


def get_hf_hub_cache_dir() -> str:
    if os.getenv("SLURM_JOB_ID") or os.getenv("LSB_JOBID"):
        cache_dir = os.getenv("HF_HUB_CACHE")
        if cache_dir is None:
            raise EnvironmentError("HF_HUB_CACHE must be set when running in a cluster environment.")
        return cache_dir

    return os.path.expanduser("~/.cache/huggingface/hub")


def find_run_dir_in_output_dir(run_dir: str):
    output_dir = get_vlm_output_dir()
    matches = list(Path(output_dir).rglob(run_dir))
    if len(matches) > 1:
        raise ValueError(f"Multiple matches found for run_dir '{run_dir}': {matches}")
    elif not matches:
        raise FileNotFoundError(f"No directory named '{run_dir}' found under {output_dir}")
    return matches[0]


def resolve_run_dir(run_dir: str | Path) -> Path:
    """Resolve an existing directory path or a run name under 'VLM_OUTPUT_DIR'."""
    path = Path(run_dir)
    if not path.is_dir() and not path.is_absolute() and len(path.parts) == 1:
        path = find_run_dir_in_output_dir(str(path))
    if not path.is_dir():
        raise FileNotFoundError(f"The specified run directory does not exist: {path}")
    return path.resolve()
