import torch.nn as nn
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Self


class VisionEncoderBase(nn.Module, ABC):

    @classmethod
    @abstractmethod
    def from_pretrained(cls, checkpoint_path: str | Path, **kwargs) -> Self:
        ...