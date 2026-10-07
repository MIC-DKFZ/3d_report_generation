"""Construct model wrappers from a run configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vlm.configs.run_config import RunConfig
    from vlm.models.abstract_vl_wrapper import AbstractVLWrapper


def create_model_wrapper(cfg: RunConfig) -> AbstractVLWrapper:
    """Construct a wrapper; the caller initializes the model and restores weights."""

    # The registry imports wrappers, which also use this helper for checkpoint loading.
    from vlm.registries.vlm_registry import VLM_REGISTRY

    try:
        model_class = VLM_REGISTRY[cfg.model.model_type].model_class
    except KeyError:
        raise ValueError(f"Unknown model type: {cfg.model.model_type}") from None

    return model_class(
        model_cfg=cfg.model,
        vision_input_type=cfg.data.vision_input_type,
        vision_embed_dim=cfg.get_vision_embed_dim(),
        vision_embed_size=cfg.get_vision_embed_size(),
        text_only=cfg.data.text_only,
        max_response_tokens=cfg.data.max_response_tokens,
        local_files_only=cfg.general.local_files_only,
    )
