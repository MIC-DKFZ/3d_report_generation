from pydantic import BaseModel
from vlm.models.owl.owl_vlm import OwlVLModel
from vlm.models.qwen25vl.qwen_vlm import QwenVLModel


class VLMConfig(BaseModel):
    """
    Configuration for a Vision-Language Model
    """
    model_id: str
    model_class: type


vlm_registry: dict[str, VLMConfig] = {
    "qwen": VLMConfig(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_class=QwenVLModel
    ),
    "owl": VLMConfig(
        model_id="mPLUG/mPLUG-Owl3-1B-241014",
        model_class=OwlVLModel
    ),
}