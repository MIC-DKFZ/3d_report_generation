"""Embedding generation: encoders, presets, multi-block inputs and the runner."""

from vlm.embeddings.registry import (
    DATASETS,
    EMBEDDINGS,
    ENCODERS,
    ROIS,
    EmbeddingPreset,
    EncoderSpec,
    register_embedding,
    register_encoder,
)
