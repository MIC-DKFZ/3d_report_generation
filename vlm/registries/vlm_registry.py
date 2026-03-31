from importlib.metadata import PackageNotFoundError, version

from pydantic import BaseModel
from vlm.models.owl.owl_vlm import OwlVLModel


TRANSFORMERS_QWEN3_CUTOFF = (4, 53, 3)


def _parse_version_triplet(raw_version: str) -> tuple[int, int, int]:
    base = raw_version.split("+", 1)[0].split("-", 1)[0]
    parts = base.split(".")
    parsed: list[int] = []

    for part in parts[:3]:
        digits = []
        for ch in part:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        parsed.append(int("".join(digits)) if digits else 0)

    while len(parsed) < 3:
        parsed.append(0)

    return parsed[0], parsed[1], parsed[2]


def _use_qwen25(version_triplet: tuple[int, int, int]) -> bool:
    major, minor, patch = version_triplet
    cutoff_major, cutoff_minor, cutoff_patch = TRANSFORMERS_QWEN3_CUTOFF

    if major != cutoff_major:
        return major < cutoff_major
    if minor != cutoff_minor:
        return minor < cutoff_minor
    return patch <= cutoff_patch

try:
    TRANSFORMERS_VERSION = _parse_version_triplet(version("transformers"))
except PackageNotFoundError:
    # Default to legacy behavior when transformers is unavailable at import time.
    TRANSFORMERS_VERSION = (0, 0, 0)

if _use_qwen25(TRANSFORMERS_VERSION):
    from vlm.models.qwen25vl.qwen_vlm import QwenVLModel
else:
    from vlm.models.qwen3vl.qwen_vlm import QwenVLModel


class VLMConfig(BaseModel):
    """
    Configuration for a Vision-Language Model
    """
    model_id: str
    model_class: type


VLM_REGISTRY: dict[str, VLMConfig] = {
    "owl": VLMConfig(
        model_id="mPLUG/mPLUG-Owl3-1B-241014",
        model_class=OwlVLModel
    )
}

if _use_qwen25(TRANSFORMERS_VERSION):
    VLM_REGISTRY["qwen"] = VLMConfig(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_class=QwenVLModel
    )
else:
    VLM_REGISTRY["qwen3"] = VLMConfig(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        model_class=QwenVLModel
    )