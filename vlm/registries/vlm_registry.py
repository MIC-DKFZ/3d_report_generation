from pydantic import BaseModel

from vlm.models.qwen3vl.wrapper import Qwen3VLWrapper


class VLMConfig(BaseModel):
    model_id: str
    model_class: type


VLM_REGISTRY: dict[str, VLMConfig] = {
    "Qwen3-VL-2B": VLMConfig(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        model_class=Qwen3VLWrapper,
    ),
}
