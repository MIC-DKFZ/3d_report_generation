import os
import shutil
import sys

from utils import logger
from configs.run_config import RunConfig
from utils.load_and_save_files import save_yaml


def setup_logging_dir(cfg: RunConfig) -> None:
    """
    Prepare the logging directory and initialize its contents:
        - input_config.yml: a copy of the provided input config file
        - final_config.yml: final merged configuration (defaults + input config (incl. base configs) + CLI args)
        - train.log: log file for runtime logs
    """
    os.makedirs(cfg.general.output_dir, exist_ok=True)
    shutil.copyfile(cfg.general.train_config, os.path.join(cfg.general.output_dir, "input_config.yml"))
    log_file_path = os.path.join(cfg.general.output_dir, "train.log")
    logger.add(log_file_path)

    # Save final run config
    final_config_path = os.path.join(cfg.general.output_dir, "final_config.yml")
    save_yaml(
        {
            "command": " ".join(sys.argv),
            "run": cfg.model_dump()
        },
        final_config_path,
        sort_keys=False
    )

    logger.info(f"Successfully set up output directory in {cfg.general.output_dir}")

