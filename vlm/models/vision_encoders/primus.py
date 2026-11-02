import inspect
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures import primus
from einops import rearrange

from vlm.models.vision_encoders.nnssl import network_kwargs, nnsslEncoderBase


class PrimusEncoder(nnsslEncoderBase):
    """
    Encoder of a Primus ViT pre-trained with nnssl, e.g. nnFoundationViT
    (https://huggingface.co/MIC-DKFZ/nnFoundationViT).

    Maps (B, C, Z, Y, X) volumes to their last-layer patch tokens as a grid
    (B, embed_dim, Z / p, Y / p, X / p); only the patch embedding and the EVA encoder are kept.
    """

    def __init__(
        self,
        network: primus.Primus,
        patch_size: tuple[int, int, int],
        target_spacing: tuple[float, float, float] | None,
        normalization: str,
    ):
        super().__init__(patch_size, target_spacing, normalization)
        self.down_projection = network.down_projection
        self.eva = network.eva
        self.embed_dim = network.embed_dim
        self.downsample_factor = tuple(network.down_projection.proj.kernel_size)

    @staticmethod
    def build_network(
        architecture_plans: dict, input_channels: int, patch_size: tuple[int, int, int]
    ) -> primus.Primus:
        return build_primus(architecture_plans, input_channels, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_projection(x)
        _, _, z, y, x_size = x.shape
        with padded_attention_head_dim():
            tokens, _ = self.eva(rearrange(x, "b c z y x -> b (z y x) c"))
        return rearrange(tokens, "b (z y x) c -> b c z y x", z=z, y=y, x=x_size)


def build_primus(
    architecture_plans: dict, input_channels: int, patch_size: tuple[int, int, int]
) -> primus.Primus:
    """
    The Primus network of an nnssl architecture plan, built like the nnU-Net and
    image_classification fine-tuning code, without dropout.

    nnssl's 'PrimusX' names the encoder's kwargs 'encoder_eva_depth' etc. and is built as
    'Primus' (the 'PrimusX' of dynamic_network_architectures takes a size name instead); the
    presets 'PrimusS/B/M/L' have no kwargs and use 8³ patches, as in nnU-Net.
    """
    class_name = architecture_plans["arch_class_name"].split(".")[-1]
    arch_kwargs = architecture_plans["arch_kwargs"] or {"patch_embed_size": (8, 8, 8)}
    if class_name == "PrimusX":
        class_name = "Primus"
        arch_kwargs = {key.removeprefix("encoder_"): value for key, value in arch_kwargs.items()}
    network_class = getattr(primus, class_name, None)
    if not (isinstance(network_class, type) and issubclass(network_class, primus.Primus)):
        raise ValueError(f"{class_name!r} is not a Primus architecture.")

    parameters = inspect.signature(network_class).parameters
    output_kwarg = "num_classes" if "num_classes" in parameters else "output_channels"
    return network_class(
        **{
            **network_kwargs(network_class, arch_kwargs),
            "input_channels": input_channels,
            "input_shape": patch_size,
            output_kwarg: 1,
        }
    )


@contextmanager
def padded_attention_head_dim(multiple: int = 8):
    """
    Zero-pad the head dim in 'F.scaled_dot_product_attention' to a multiple of 'multiple'.

    Head dims such as 66 (nnFoundationViT) have no fused kernel, and the math fallback needs
    12 GiB per sample in float32. Zero padding leaves the result unchanged; the padded output
    channels are dropped.
    """
    sdpa = F.scaled_dot_product_attention

    def padded_sdpa(query, key, value, *args, scale=None, **kwargs):
        head_dim = query.shape[-1]
        pad = -head_dim % multiple
        if not pad:
            return sdpa(query, key, value, *args, scale=scale, **kwargs)
        query, key, value = (F.pad(t, (0, pad)) for t in (query, key, value))
        scale = head_dim**-0.5 if scale is None else scale
        return sdpa(query, key, value, *args, scale=scale, **kwargs)[..., :head_dim]

    F.scaled_dot_product_attention = padded_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = sdpa
