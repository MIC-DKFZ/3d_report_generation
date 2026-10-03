"""Dataloaders shared by inference and training-time evaluation."""

from collections.abc import Callable

from torch.utils.data import DataLoader, Dataset

from vlm.data.samplers import EvalDistributedSampler


def create_eval_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    collate_fn: Callable,
    num_workers: int,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    sampler = (
        EvalDistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if is_distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        # Pinned memory caused long stalls during training callback evaluation.
        pin_memory=False,
    )
