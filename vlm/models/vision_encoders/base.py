from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import torch.nn as nn


class VisionEncoderBase(nn.Module, ABC):
    # Preprocessing the encoder expects, used to encode new images (see 'inference_nifti.py'),
    # set per class or by 'from_pretrained': the voxel spacing (z, y, x) in mm (None: no
    # resampling) and the intensity normalization scheme of
    # 'vlm.data.preprocessing.encoder_input.apply_normalization'.
    target_spacing: tuple[float, float, float] | None = None
    normalization: str | None = None

    @classmethod
    @abstractmethod
    def from_pretrained(cls, checkpoint_path: str | Path, **kwargs) -> Self: ...
