"""ToothFairy4 image and embedding paths."""

from pathlib import Path

from vlm.utils.data.vision_input_type import VisionInputType

from .base import MedicalImageDataset


class ToothFairy4Dataset(MedicalImageDataset):
    """
    ToothFairy4 dataset - dental CBCT scans with reports.
    """

    def _get_vision_input_path(self, image_id) -> Path:
        if self.vision_input_type == VisionInputType.IMAGE:
            return self.vision_input_path / f"{image_id}.nii.gz"
        elif self.vision_input_type == VisionInputType.EMBEDDING:
            return self.vision_input_path / f"{image_id}{self.image_id_suffix}_embedding.bl2"
        else:
            raise ValueError(f"Unsupported vision input type: {self.vision_input_type}")
