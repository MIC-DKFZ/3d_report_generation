from pydantic import BaseModel

from vlm.models.gemma3.wrapper import Gemma3TextVLWrapper
from vlm.models.phi4mini.wrapper import Phi4MiniVLWrapper
from vlm.models.qwen25vl.wrapper import Qwen25VLWrapper
from vlm.models.qwen3.wrapper import Qwen3TextVLWrapper
from vlm.models.qwen3vl.wrapper import Qwen3VLWrapper


class VLMConfig(BaseModel):
    model_id: str
    model_class: type


VLM_REGISTRY: dict[str, VLMConfig] = {
    "Qwen2.5-VL-3B": VLMConfig(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_class=Qwen25VLWrapper,
    ),
    "Qwen3-1.7B": VLMConfig(
        model_id="Qwen/Qwen3-1.7B",
        model_class=Qwen3TextVLWrapper
    ),
    "Qwen3-VL-2B": VLMConfig(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        model_class=Qwen3VLWrapper
    ),
    "Qwen3-VL-4B": VLMConfig(
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        model_class=Qwen3VLWrapper
    ),
    "Qwen3-VL-8B": VLMConfig(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        model_class=Qwen3VLWrapper
    ),
    "Phi4Mini": VLMConfig(
        model_id="microsoft/Phi-4-mini-instruct",
        model_class=Phi4MiniVLWrapper
    ),
    "Gemma3-4B": VLMConfig(
        model_id="google/gemma-3-4b-it",
        model_class=Gemma3TextVLWrapper
    ),
}
