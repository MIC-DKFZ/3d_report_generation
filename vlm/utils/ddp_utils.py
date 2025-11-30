from collections import defaultdict
from itertools import chain

import torch
import os
from vlm.utils import logger
from torch import distributed as dist


def gather_inference_outputs(local_results: list | dict, main_rank: int = 0) -> list | dict[str, list] | None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Prepare the destination only on the main process
    if rank == main_rank:
        all_results = [None for _ in range(world_size)]
    else:
        all_results = None

    # Gather lists
    dist.gather_object(local_results, all_results, dst=main_rank)

    # Concatenate on the main process
    if rank == main_rank:
        if isinstance(local_results, list):
            merged_list: list = list(chain.from_iterable(all_results))  # type: ignore
            return merged_list
        elif isinstance(local_results, dict):
            merged_dict: dict[str, list] = defaultdict(list)
            for d in all_results:
                for k, lis in d.items():
                    merged_dict[k].extend(lis)
            return merged_dict
        else:
            raise ValueError(f"'local_results' must be dict or list: {type(local_results)}")
    else:
        return None


def setup_ddp() -> tuple[bool, int, int, int, int]:
    """
    Set up distributed training environment.
    This function handles the distributed training initialization as recommended in the JSC tutorial.
    """
    # Check if we're in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE"))

        logger.info(f"DDP detected: rank={rank}, world_size={world_size}, "
                    f"local_rank={local_rank}, local_world_size={local_world_size}")

        # Set the device for this process
        torch.cuda.set_device(local_rank)

        # Initialize the process group
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        logger.info(f"Process group initialized on rank {rank}")

        return True, rank, world_size, local_rank, local_world_size
    else:
        logger.info("Single GPU/CPU detected")
        return False, 0, 1, 0, 1
