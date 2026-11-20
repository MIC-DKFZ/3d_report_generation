"""
Encoders and embedding presets of the embedding generation
(vlm/scripts/embeddings/generate_embeddings.py).

An encoder ('EncoderSpec') says how to build a vision encoder and what input it takes. An
embedding preset ('EmbeddingPreset') names one set of embeddings: an encoder, its checkpoint and,
per dataset, the ROI and processing mode. The public encoders and presets are registered here;
other packages add theirs with 'register_encoder' and 'register_embedding' before calling the CLI.
"""

from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn

from vlm.configs.data_config import validate_orientation
from vlm.data.preprocessing.encoder_input import NORMALIZATIONS, PROCESSING_MODES
from vlm.embeddings.multiblock import Shape
from vlm.models.vision_encoders.primus import PrimusEncoder
from vlm.models.vision_encoders.resenc import ResEncEncoder

Spacing = tuple[float, float, float]

# Orientations are those the datasets are mostly stored in, so that most volumes stay unchanged.
DATASETS = {
    "ctrate": {"orientation": "LPS", "split_folders": ["train_fixed", "valid_fixed"]},
    "merlin": {"orientation": "LAS", "split_folders": [""]},
}

# ROIs from TotalSegmentator 'total' masks, as recorded in 'data.vision_input_preprocessing'.
# A 'roi_mask' starting with "TotalSegmentator 'total'" lets the NIfTI inference generate masks.
ROIS = {
    "lungs": {
        "roi_mask": "TotalSegmentator 'total', lungs (labels 10-14)",
        "mask_label_ids": [10, 11, 12, 13, 14],
        "enforce_min_bbox": False,
        "outlier_method": "none",
    },
    "foreground": {
        "roi_mask": "TotalSegmentator 'total', foreground of all labels",
        "mask_label_ids": None,
        "enforce_min_bbox": True,
        "outlier_method": "connected_components",
        "outlier_min_voxels": 400,
    },
}


@dataclass(frozen=True)
class EncoderSpec:
    """
    'build' maps a checkpoint (file, Hugging Face repository or None) to a module that maps
    (B, 1, *input_shape) inputs to (B, C, d, h, w) embeddings. 'input_shape' None takes the
    module's 'patch_size'. 'orientation' (e.g. 'RAS') replaces the dataset's for encoders
    pretrained on one orientation. 'embedding_encoder' is the class in vlm/models/vision_encoders that
    loads the same checkpoint, recorded so that inference on new volumes encodes them the same
    way; None if there is none.
    """

    build: Callable[[str | None], nn.Module]
    target_spacing: Spacing | None
    normalization: str
    input_shape: Shape | None = None
    orientation: str | None = None
    embedding_encoder: str | None = None

    def __post_init__(self):
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"Unknown normalization {self.normalization!r}")
        if self.orientation is not None:
            validate_orientation(self.orientation)


@dataclass(frozen=True)
class EmbeddingPreset:
    """
    'rois' per dataset (None: the whole volume) and 'processing_modes' per dataset (one mode for
    all datasets if a string). 'grid' (gz, gy, gx) > 1 embeds a window of that many encoder inputs
    block-wise and merges the block embeddings. 'normalization' replaces the encoder's.
    'checkpoint' None: the checkpoint is a local file, passed with '--checkpoint'.
    """

    encoder: str
    checkpoint: str | None
    rois: dict[str, str] | None
    processing_modes: dict[str, str] | str
    grid: Shape = (1, 1, 1)
    normalization: str | None = None
    description: str = field(default="", compare=False)

    def processing_mode(self, dataset: str) -> str:
        if isinstance(self.processing_modes, str):
            return self.processing_modes
        return self.processing_modes[dataset]

    def roi(self, dataset: str) -> str | None:
        return None if self.rois is None else self.rois[dataset]


ENCODERS: dict[str, EncoderSpec] = {}
EMBEDDINGS: dict[str, EmbeddingPreset] = {}


def register_encoder(name: str, spec: EncoderSpec) -> None:
    if name in ENCODERS and ENCODERS[name] != spec:
        raise ValueError(f"Encoder {name!r} is already registered differently.")
    ENCODERS[name] = spec


def register_embedding(name: str, preset: EmbeddingPreset) -> None:
    if name in EMBEDDINGS and EMBEDDINGS[name] != preset:
        raise ValueError(f"Embedding {name!r} is already registered differently.")
    if preset.encoder not in ENCODERS:
        raise ValueError(f"Embedding {name!r} uses the unregistered encoder {preset.encoder!r}.")
    modes = preset.processing_modes
    for mode in [modes] if isinstance(modes, str) else modes.values():
        if mode not in PROCESSING_MODES:
            raise ValueError(f"Embedding {name!r}: unknown processing mode {mode!r}.")
    for roi in (preset.rois or {}).values():
        if roi not in ROIS:
            raise ValueError(f"Embedding {name!r}: unknown ROI {roi!r}.")
    EMBEDDINGS[name] = preset


register_encoder(
    "primus",
    EncoderSpec(
        PrimusEncoder.from_pretrained,
        (1.0, 1.0, 1.0),
        "z_score",
        embedding_encoder="PrimusEncoder",
    ),
)
register_encoder(
    "resenc",
    EncoderSpec(
        ResEncEncoder.from_pretrained,
        (1.0, 1.0, 1.0),
        "z_score",
        embedding_encoder="ResEncEncoder",
    ),
)

PAPER_ROIS = {"ctrate": "lungs", "merlin": "foreground"}

# nnFoundation paper
register_embedding(
    "nnFoundationViT",
    EmbeddingPreset(
        "primus",
        "MIC-DKFZ/nnFoundationViT",
        PAPER_ROIS,
        {"ctrate": "center_crop", "merlin": "resized"},
        description="nnFoundation paper; ROI center cropped (CT-RATE) or resized (Merlin).",
    ),
)
register_embedding(
    "nnFoundationCNN",
    EmbeddingPreset(
        "resenc",
        "MIC-DKFZ/nnFoundationCNN",
        PAPER_ROIS,
        "resized",
        description="nnFoundation paper; ROI resized.",
    ),
)
# Released nnFoundation models
register_embedding(
    "nnFoundationViT_release",
    EmbeddingPreset(
        "primus",
        "MIC-DKFZ/nnFoundationViT@c691f2115140458ec64017712eb8ebbc8dde6ae4",
        {"ctrate": "foreground", "merlin": "foreground"},
        "resized_direct",
        description="Released nnFoundationViT models; foreground resized with anti-aliasing.",
    ),
)
# RMR paper
register_embedding(
    "Primus-1B_ROI",
    EmbeddingPreset(
        "primus",
        "MIC-DKFZ/PrimusM-MAE",
        PAPER_ROIS,
        "resized",
        description="RMR paper; one input block from the ROI.",
    ),
)
register_embedding(
    "Primus-1B_Whole",
    EmbeddingPreset(
        "primus",
        "MIC-DKFZ/PrimusM-MAE",
        None,
        "resized",
        description="RMR paper; one input block from the whole volume.",
    ),
)
for _grid in ((2, 1, 2), (1, 2, 2)):
    register_embedding(
        f"Primus-4B_{'_'.join(map(str, _grid))}_ROI",
        EmbeddingPreset(
            "primus",
            "MIC-DKFZ/PrimusM-MAE",
            PAPER_ROIS,
            "resized",
            grid=_grid,
            description=f"RMR paper; four input blocks in a {' × '.join(map(str, _grid))} grid "
            "from the ROI.",
        ),
    )
