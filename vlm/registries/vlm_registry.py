from pydantic import BaseModel
from vlm.models.owl.owl_vlm import OwlVLModel
from vlm.models.qwen25vl.qwen_vlm import QwenVLModel
from vlm.models.qwen3vl.qwen_vlm import QwenVLModel as Qwen3VLModel


class VLMConfig(BaseModel):
    """
    Configuration for a Vision-Language Model
    """
    model_id: str
    model_class: type


VLM_REGISTRY: dict[str, VLMConfig] = {
    "qwen": VLMConfig(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_class=QwenVLModel
    ),
    "owl": VLMConfig(
        model_id="mPLUG/mPLUG-Owl3-1B-241014",
        model_class=OwlVLModel
    ),
    "qwen3": VLMConfig(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        model_class=Qwen3VLModel
    )
}