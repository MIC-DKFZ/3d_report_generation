import math
import pydoc

import torch
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from torch import nn

from vlm.models.vision_encoders.nnssl import network_kwargs, nnsslEncoderBase

# nnssl's 'ResEncL' preset (nnssl.architectures.architecture_registry.get_res_enc_l), the
# architecture of nnFoundationCNN.
RESENC_L_KWARGS = {
    "n_stages": 6,
    "features_per_stage": [32, 64, 128, 256, 320, 320],
    "conv_op": nn.Conv3d,
    "kernel_sizes": [[3, 3, 3]] * 6,
    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
    "conv_bias": True,
    "norm_op": nn.InstanceNorm3d,
    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
    "nonlin": nn.LeakyReLU,
    "nonlin_kwargs": {"inplace": True},
}


class ResEncEncoder(nnsslEncoderBase):
    """
    Encoder of a residual encoder U-Net (ResEnc) pre-trained with nnssl, e.g. nnFoundationCNN
    (https://huggingface.co/MIC-DKFZ/nnFoundationCNN).

    Maps (B, C, Z, Y, X) volumes to the feature map of the last encoder stage
    (B, features_per_stage[-1], Z / f, Y / f, X / f), with f the product of the stage strides;
    the decoder is dropped.
    """

    def __init__(
        self,
        network: ResidualEncoderUNet,
        patch_size: tuple[int, int, int],
        target_spacing: tuple[float, float, float] | None,
        normalization: str,
    ):
        super().__init__(patch_size, target_spacing, normalization)
        self.encoder = network.encoder
        self.encoder.return_skips = False
        self.embed_dim = self.encoder.output_channels[-1]
        self.downsample_factor = tuple(map(math.prod, zip(*self.encoder.strides)))

    @staticmethod
    def build_network(
        architecture_plans: dict, input_channels: int, patch_size: tuple[int, int, int]
    ) -> ResidualEncoderUNet:
        return build_resenc(architecture_plans, input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_resenc(architecture_plans: dict, input_channels: int) -> ResidualEncoderUNet:
    """
    The ResEnc U-Net of an nnssl architecture plan, without dropout: the 'ResEncL' preset, or a
    'ResidualEncoderUNet' with its kwargs, whose 'arch_kwargs_requiring_import' are class paths.
    """
    class_name = architecture_plans["arch_class_name"].split(".")[-1]
    if class_name == "ResEncL":
        arch_kwargs = RESENC_L_KWARGS
    elif class_name == "ResidualEncoderUNet":
        arch_kwargs = dict(architecture_plans["arch_kwargs"])
        for key in architecture_plans.get("arch_kwargs_requiring_import") or ():
            if arch_kwargs[key] is not None:
                arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
    else:
        raise ValueError(f"{class_name!r} is not a ResEnc architecture.")
    return ResidualEncoderUNet(
        **network_kwargs(ResidualEncoderUNet, arch_kwargs),
        input_channels=input_channels,
        num_classes=1,
        deep_supervision=False,
    )
