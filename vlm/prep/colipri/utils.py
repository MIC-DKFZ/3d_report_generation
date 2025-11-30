from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from typing import get_args

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.architectures.primus import Eva
from dynamic_network_architectures.architectures.primus import Primus
from einops.layers.torch import Rearrange
from torch import nn
from torch._dynamo import OptimizedModule

from .adaptation_plan import DYN_ARCHITECTURE_PRESETS
from .adaptation_plan import AdaptationPlan
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.get_network_from_plan import get_network_from_plans


def convert_to_vision_only_forward_module(
    network: nn.Module, adapt_plan: AdaptationPlan, is_ddp: bool = False
) -> tuple[nn.Module, int, dict[int, str]]:
    """
    Takes any module

    Args:
        network (nn.Module): The network to convert
        adapt_plan (AdaptationPlan): The adaptation plan of the current trainer
        is_ddp (bool, optional): Whether the network is wrapped in a DDP module. Defaults to False.

    Returns:
        tuple[nn.Module, int]: The vision only forward module, the embedding dimensions and a dict mapping probe locations to their module path
    """

    module = network
    if isinstance(module, OptimizedModule):  # If we compiled the network we add the module prefix
        module = module._orig_mod

    if is_ddp:  # need to get the original module from ddp wrapper
        module = module.module

    encoder = module.get_submodule(adapt_plan.key_to_encoder)
    if isinstance(encoder, Eva):
        # If we have an Eva style encoder, we know the projection layer is not in the Eva block
        #   So we grab that module from outside.
        proj_module = module.get_submodule(adapt_plan.keys_to_in_proj[0])

        probe_locations = {
            cnt: f"{deepcopy(adapt_plan.key_to_encoder).replace('.', ':')}.blocks.{i}"
            for cnt, i in enumerate(range(len(encoder.get_submodule("blocks"))))
        }
        # We create a new submodule, which holds the encoder and it's down projection
        # This allows all forward calls to be made through that model not requiring specific data

        # 3DConv -> Rearrange -> Encoder
        network = nn.Sequential(
            OrderedDict(
                [
                    (
                        deepcopy(adapt_plan.keys_to_in_proj[0]).replace(".", ":"),
                        proj_module,
                    ),
                    ("rearrange", Rearrange("b c w h d -> b (w h d) c")),
                    ((deepcopy(adapt_plan.key_to_encoder).replace(".", ":")), encoder),
                ]
            )
        )

        embed_dim: int = encoder.embed_dim
    else:
        raise NotImplementedError("Only Primus supported for now")
    return network, embed_dim, probe_locations


def build_network_from_adaptation_plan(
    adapt_plan: AdaptationPlan,
) -> AbstractDynamicNetworkArchitectures:
    """
    Builds a network from the adaptation plan.
    """
    if adapt_plan.architecture_plans.arch_class_name in get_args(DYN_ARCHITECTURE_PRESETS):
        downstream_arch = get_network_from_plans(
            adapt_plan.architecture_plans.arch_class_name,
            arch_kwargs=asdict(adapt_plan.architecture_plans.arch_kwargs),
            arch_kwargs_req_import=adapt_plan.architecture_plans.arch_kwargs_requiring_import,
            input_channels=adapt_plan.pretrain_num_input_channels,
            output_channels=2,  # Some arbitrary choice
            deep_supervision=False,
            allow_init=False,  # Will be loaded from pre-trained weights, so does not matter!
        )
    else:
        downstream_arch = get_network_by_name(
            adapt_plan.recommended_downstream_patchsize,
            adapt_plan.architecture_plans.arch_class_name,
            # This assures we can load the same weights -- May not be transferrable downstream because diff. in channels)
            adapt_plan.pretrain_num_input_channels,
            2,  # Number of output channels -- Does not matter (like e.g. decoder)
            encoder_only=False,
            deep_supervision=False,
            arch_kwargs=None,
        )
    return downstream_arch


def get_probe_locations_of_arch(
    network: AbstractDynamicNetworkArchitectures,
) -> dict[int, str]:
    """
    Returns the probe locations from the adaptation plan.
    """

    if isinstance(network, Primus):
        # If we have an Eva style encoder, we know the projection layer is not in the Eva block
        #   So we grab that module from outside.
        key_to_encoder = network.key_to_encoder
        probe_locations = {
            cnt: f"{deepcopy(key_to_encoder)}.blocks.{i}"
            for cnt, i in enumerate(range(len(network.get_submodule("eva.blocks"))))
        }
    else:
        raise NotImplementedError("Only Primus supported for now")
    return probe_locations
