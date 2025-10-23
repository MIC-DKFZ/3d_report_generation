import math
from collections.abc import Iterator
from typing import Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


__all__ = ["EvalDistributedSampler"]


_T_co = TypeVar("_T_co", covariant=True)


class EvalDistributedSampler(Sampler[_T_co]):
    """
    A modified version of 'torch.utils.data.DistributedSampler' for evaluation and inference.

    Unlike the default sampler, this implementation does not pad the dataset to make it evenly
    divisible across replicas. Instead, it splits all samples across ranks as evenly as possible,
    ensuring that each sample is processed exactly once and that no duplicates or padding occur.
    Some ranks may receive one extra sample if the dataset size is not perfectly divisible.
    """
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        # drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        # self.drop_last = drop_last
        # # If the dataset length is evenly divisible by # of replicas, then there
        # # is no need to drop any data, since the dataset will be split equally.
        # if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
        #     # Split to nearest available length that is evenly divisible.
        #     # This is to ensure each rank receives the same amount of data when
        #     # using this Sampler.
        #     self.num_samples = math.ceil(
        #         (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
        #     )
        # else:
        #     self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        # self.total_size = self.num_samples * self.num_replicas

        self.total_size = len(self.dataset)  # type: ignore[arg-type]
        self.num_samples = math.ceil((self.total_size - self.rank) / self.num_replicas)

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # if not self.drop_last:
        #     # add extra samples to make it evenly divisible
        #     padding_size = self.total_size - len(indices)
        #     if padding_size <= len(indices):
        #         indices += indices[:padding_size]
        #     else:
        #         indices += (indices * math.ceil(padding_size / len(indices)))[
        #             :padding_size
        #         ]
        # else:
        #     # remove tail of data to make it evenly divisible.
        #     indices = indices[: self.total_size]
        # if len(indices) != self.total_size:
        #     raise AssertionError(
        #         f"Number of indices ({len(indices)}) does not match total_size ({self.total_size})"
        #     )

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(indices)}) does not match num_samples ({self.num_samples})"
            )

        # pyrefly: ignore  # bad-return
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch