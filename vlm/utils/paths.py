import os
from pathlib import Path
from vlm.utils import logger


def get_env_var(name: str, required: bool = True, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        msg = f"[Environment Variables] {name} is not defined!"
        if required:
            raise EnvironmentError(msg)
        logger.warning_main(msg)
    return value


VLM_CONFIG_DIR = get_env_var("VLM_CONFIG_DIR")
VLM_OUTPUT_DIR = get_env_var("VLM_OUTPUT_DIR")


def get_hf_hub_cache_dir() -> str:
    if os.getenv("SLURM_JOB_ID") or os.getenv("LSB_JOBID"):
        cache_dir = os.getenv("HF_HUB_CACHE")
        if cache_dir is None:
            raise EnvironmentError("HF_HUB_CACHE must be set when running in a cluster environment.")
        return cache_dir

    return os.path.expanduser("~/.cache/huggingface/hub")


def find_run_dir_in_output_dir(run_dir: str):
    matches = list(Path(VLM_OUTPUT_DIR).rglob(run_dir))
    if len(matches) > 1:
        raise ValueError(f"Multiple matches found for run_dir '{run_dir}': {matches}")
    elif not matches:
        raise FileNotFoundError(f"No directory named '{run_dir}' found under {VLM_OUTPUT_DIR}")
    return matches[0]
