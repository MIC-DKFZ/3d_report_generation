import inspect
import math
from abc import abstractmethod
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from vlm.models.vision_encoders.base import VisionEncoderBase
from vlm.utils.paths import get_hf_hub_cache_dir

# Normalization schemes of nnssl plans and their names in 'encoder_input.apply_normalization'.
NORMALIZATION_SCHEMES = {"ZScoreNormalization": "z_score"}


class nnsslEncoderBase(VisionEncoderBase):
    """
    Encoder of a network pre-trained with nnssl, e.g. nnFoundationViT or nnFoundationCNN
    (https://huggingface.co/MIC-DKFZ).

    The architecture and the preprocessing come from the checkpoint's nnssl adaptation plan.
    Subclasses build the pre-training network from the plan, keep its stem and encoder (the
    plan's 'key_to_stem' and 'key_to_encoder' modules, under the same names) and map
    (B, C, Z, Y, X) volumes to a grid (B, embed_dim, Z / f, Y / f, X / f), with f the
    'downsample_factor'.
    """

    checkpoint_file = "checkpoint_final.pth"

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        target_spacing: tuple[float, float, float] | None,
        normalization: str,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.target_spacing = target_spacing
        self.normalization = normalization

    @staticmethod
    @abstractmethod
    def build_network(
        architecture_plans: dict, input_channels: int, patch_size: tuple[int, int, int]
    ) -> nn.Module:
        """The pre-training network of an nnssl architecture plan."""

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path | None, **kwargs) -> Self:
        """
        Load the encoder from an nnssl checkpoint file, or from a Hugging Face hub repository
        (ID or URL) holding one as 'checkpoint_final.pth', at a revision if given as
        '<repository>@<revision>' (e.g. a commit hash, as exported checkpoints record it).

        The encoder input is the plan's 'recommended_downstream_patchsize', as in nnU-Net and
        image_classification; a learned positional embedding ('key_to_lpe') is resized from
        the pre-training patch size to it.
        """
        if checkpoint_path is None:
            raise ValueError(f"{cls.__name__} needs a checkpoint file or Hugging Face repository.")
        if not Path(checkpoint_path).is_file():
            repo_id = str(checkpoint_path).removeprefix("https://huggingface.co/").strip("/")
            repo_id, _, revision = repo_id.partition("@")
            checkpoint_path = hf_hub_download(
                repo_id,
                cls.checkpoint_file,
                revision=revision or None,
                cache_dir=get_hf_hub_cache_dir(),
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        plan = checkpoint["nnssl_adaptation_plan"]
        (pretrain_config,) = plan["pretrain_plan"]["configurations"].values()
        pretrain_patch_size = tuple(pretrain_config["patch_size"])
        patch_size = tuple(plan["recommended_downstream_patchsize"])

        network = cls.build_network(
            plan["architecture_plans"], plan["pretrain_num_input_channels"], patch_size
        )
        encoder = cls(network, patch_size, **pretrain_preprocessing(pretrain_config), **kwargs)
        stem_and_encoder = (f"{plan['key_to_stem']}.", f"{plan['key_to_encoder']}.")
        weights = {
            key: value
            for key, value in checkpoint["network_weights"].items()
            if key.startswith(stem_and_encoder)
        }
        key_to_lpe = plan.get("key_to_lpe")
        if key_to_lpe is not None and pretrain_patch_size != patch_size:
            weights[key_to_lpe] = resize_pos_embed(
                weights[key_to_lpe],
                *(
                    tuple(s // f for s, f in zip(size, encoder.downsample_factor))
                    for size in (pretrain_patch_size, patch_size)
                ),
            )
        encoder.load_state_dict(weights)
        return encoder.eval()


def pretrain_preprocessing(pretrain_config: dict) -> dict:
    """Target spacing (None: no resampling) and normalization of an nnssl pre-training config."""
    schemes = set(pretrain_config["normalization_schemes"])
    if len(schemes) != 1 or not schemes <= NORMALIZATION_SCHEMES.keys():
        raise ValueError(f"Unsupported normalization schemes: {sorted(schemes)}")
    spacing = pretrain_config["spacing"]
    no_resampling = pretrain_config["spacing_style"] == "noresample" or spacing is None
    return {
        "target_spacing": None if no_resampling else tuple(spacing),
        "normalization": NORMALIZATION_SCHEMES[schemes.pop()],
    }


def resize_pos_embed(
    pos_embed: torch.Tensor, source_grid: tuple[int, ...], target_grid: tuple[int, ...]
) -> torch.Tensor:
    """
    Trilinearly resize a (1, prefix + prod(grid), C) positional embedding to another token
    grid, as in nnU-Net and image_classification; prefix tokens (e.g. registers) are kept.
    """
    num_prefix_tokens = pos_embed.shape[1] - math.prod(source_grid)
    prefix, patches = pos_embed[:, :num_prefix_tokens], pos_embed[:, num_prefix_tokens:]
    patches = patches.transpose(1, 2).reshape(1, -1, *source_grid)
    patches = F.interpolate(patches.float(), size=target_grid, mode="trilinear")
    patches = patches.flatten(2).transpose(1, 2).to(pos_embed.dtype)
    return torch.cat((prefix, patches), dim=1)


def network_kwargs(network_class: type, kwargs: dict) -> dict:
    """The kwargs 'network_class' takes, without dropout and with lists as tuples."""
    parameters = inspect.signature(network_class).parameters
    return {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in kwargs.items()
        if key in parameters and "drop" not in key
    }
