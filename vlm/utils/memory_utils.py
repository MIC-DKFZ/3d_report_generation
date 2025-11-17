"""Memory utilities for monitoring and managing GPU memory during training."""

import gc
import torch
from vlm.utils import logger


def log_memory_usage(stage=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - reserved

        logger.info(
            f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB"
        )
    else:
        logger.info(f"[{stage}] CUDA not available")


def aggressive_cleanup():
    """Perform aggressive memory cleanup."""
    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Additional cleanup
    import psutil

    process = psutil.Process()
    logger.info(f"Process memory usage: {process.memory_info().rss / 1024**3:.2f}GB")


def monitor_memory_during_training():
    """Context manager for monitoring memory during training steps."""

    class MemoryMonitor:
        def __enter__(self):
            log_memory_usage("Before Training Step")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            log_memory_usage("After Training Step")
            if exc_type is not None:
                logger.error_main(f"Exception during training: {exc_val}")
                aggressive_cleanup()

    return MemoryMonitor()


def setup_memory_efficient_training():
    """Setup memory-efficient training settings."""
    if torch.cuda.is_available():
        # Enable memory fraction for CUDA
        torch.cuda.set_per_process_memory_fraction(0.95)

        # Set memory growth
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except AttributeError:
            pass

        logger.info_main("Memory efficient training settings enabled")
