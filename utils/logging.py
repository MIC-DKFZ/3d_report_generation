import os
import sys
import inspect
import logging
from pathlib import Path
from typing import Protocol, cast, Union
from loguru import logger as loguru_logger

import torch.distributed as dist


def rank_zero_filter(record):
    try:
        ddp_available = dist.is_available() and dist.is_initialized()
        if ddp_available:
            return int(os.environ["RANK"]) == 0
        else:
            return True
    except Exception: # Fallback for non-DDP or uninitialized DDP
        return True

def filter_evaluate_loading(record):
    """Remove unnecessary warnings from evaluate.loading"""
    return not record["name"].startswith("evaluate.loading")


def log_banner(text: str, width: int = 100):
    line = f" {text} ".center(width, "=")
    logger.info_main("\n\n" + line + "\n")


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level: str | int = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame = inspect.currentframe()
        depth = 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logger(log_file_path: str | Path | None = None):
    loguru_logger.remove()

    if "LOSS" not in [lvl.name for lvl in loguru_logger._core.levels.values()]:
        loguru_logger.level(
            "LOSS",
            no=21,
            color="<magenta><b>",
        )

    rank = int(os.getenv("RANK", "0"))
    rank_color = "green" if rank == 0 else "white"

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        f"<{rank_color}><b>RANK {{extra[rank]}}</b></{rank_color}> | "  # add rank column to default format
        "<level>{message}</level>"
    )
    if log_file_path is not None:
        loguru_logger.add(str(log_file_path), enqueue=True, format=fmt, level="DEBUG")
    loguru_logger.add(sys.stderr, enqueue=True, format=fmt, colorize=True)

    loguru_logger.configure(extra={"rank": rank})

    root_logger = logging.getLogger()
    if not any(isinstance(h, InterceptHandler) for h in root_logger.handlers):
        logging.basicConfig(handlers=[InterceptHandler()], level=logging.WARNING)

    loguru_logger.info(f"Logger ready on rank {rank}")


class MainRankLoggingProtocol(Protocol):
    def info_main(self, msg: str, *args, **kwargs) -> None: ...
    def warning_main(self, msg: str, *args, **kwargs) -> None: ...
    def error_main(self, msg: str, *args, **kwargs) -> None: ...

def _build_main_rank_logger(_level: str):
    is_main = int(os.environ.get("RANK", "0")) == 0  # is_main_process() -> circular import
    def _log_main(msg, *a, **kw):
        if is_main:
            getattr(loguru_logger, _level)(msg, *a, **kw)
    return _log_main

for level in ["info", "warning", "error"]:
    setattr(loguru_logger, f"{level}_main", _build_main_rank_logger(level))


# Use this logger to retain all standard Loguru methods while adding the
# extra main-rank-only methods defined in MainRankLoggingProtocol.
logger = cast(Union[type(loguru_logger), MainRankLoggingProtocol], loguru_logger)













