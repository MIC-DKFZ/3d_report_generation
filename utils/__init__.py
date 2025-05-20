import os
import inspect
import logging
from types import FrameType
import torch.distributed as dist

from loguru import logger

logger.remove()


def rank_zero_filter(record):
    try:
        ddp_available = dist.is_available() and dist.is_initialized()
        if ddp_available:
            return int(os.environ["RANK"]) == 0
        else:
            return True
    except Exception: # Fallback for non-DDP or uninitialized DDP
        return True


logger.level(
    "LOSS",
    no=21,
    color="<magenta><b>",
)


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level: str | int = logger.level(record.levelname).name
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

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[InterceptHandler()], level=logging.WARNING)
