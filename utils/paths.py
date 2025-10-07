import os
from utils import logger


def get_env_var(name: str, required: bool = True, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        msg = f"[Environment Variables] {name} is not defined!"
        if required:
            raise EnvironmentError(msg)
        logger.warning(msg)
    return value


VLM_CONFIG_DIR = get_env_var("VLM_CONFIG_DIR")
VLM_OUTPUT_DIR = get_env_var("VLM_OUTPUT_DIR")
