from models.owl_vlm import OwlVLModel
from models.qwen_vlm import QwenVLModel
from utils.config_models import VLMConfig

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