import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.architectures.primus import Primus
from einops.layers.torch import Rearrange
from .pretrainedTrainer import PretrainedTrainer
from torch import nn

from .adaptation_plan import AdaptationPlan
from .adaptation_planning_utils import mb_load_ckpt
from .attention_pooling import AttentionPool1D
from .utils import build_network_from_adaptation_plan
from .plan import ConfigurationPlan


def get_vision_pooling_class_from_ckpt_path(
    ckpt_path: str, txt_encoder_embed_dim: int, txt_encoder_heads: int
) -> type[nn.Module]:
    # Get the vision pooling class based on the checkpoint path.
    return AttentionPool1D(embed_dim=txt_encoder_embed_dim, num_heads=txt_encoder_heads)  # Default


def build_dense_embedding_network(ckpt_path: str | dict, adaptation_plan_path: str, load_weights: bool = True):
    """
    Builds a dense embedding network from a checkpoint path and an adaptation plan.

    Args:
        ckpt_path (str | dict): Path or state_dict of the checkpoint file with the weights.
        adaptation_plan_path (str): Path to the adaptation plan JSON file.
        load_weights (bool, optional): Flag if to load the weights from the checkpoint. Defaults to True.

    Returns:
        _type_: network with loaded (or without loaded) weights.
    """
    adapt_plan = AdaptationPlan.from_dict(load_json(adaptation_plan_path))
    # Get configuration from pretraining plan
    ssl_config: ConfigurationPlan = list(adapt_plan.pretrain_plan.configurations.values())[0]

    network: AbstractDynamicNetworkArchitectures
    network = build_network_from_adaptation_plan(adapt_plan).cpu()  # For loading
    assert isinstance(network, Primus), "Only Primus architectures are currently supported for classification probes."
    if (adapt_plan.architecture_plans.arch_kwargs is not None) and not adapt_plan.architecture_plans.arch_kwargs.use_abs_pos_embed:
        network.key_to_lpe = None  # Disable LPE if not used in pretraining
    if load_weights:
        PretrainedTrainer.load_pretrained_weights(
            network,
            ckpt_path,
            pt_input_channels=adapt_plan.pretrain_num_input_channels,
            downstream_input_channels=1,  #
            pt_input_patchsize=ssl_config.patch_size,
            downstream_input_patchsize=ssl_config.patch_size,
            pt_key_to_encoder=adapt_plan.key_to_encoder,
            pt_keys_to_in_proj=adapt_plan.keys_to_in_proj,
            pt_key_to_lpe=adapt_plan.key_to_lpe,
            pt_key_to_stem=adapt_plan.key_to_stem,
        )
    return network.cuda()


def get_temperature_from_ckpt_path(ckpt_path: str) -> float:
    """
    Tries to extract the temperature value from the checkpoint path.

    Args:
        ckpt_path (str): Path to the checkpoint file or already loaded checkpoint. Has to contain stored weights.

    Raises:
        ValueError: When multiple temperature values are found in the state dict.

    Returns:
        float: The temperature value extracted from the checkpoint state dict.
    """
    state_dict = mb_load_ckpt(ckpt_path, network_key="loss_state_dict")
    state_dict = {k: v for k, v in state_dict.items() if k.startswith("temperature")}
    if len(state_dict) == 0:
        return 0.07
    elif len(state_dict) == 1:
        return float(state_dict["temperature"].item())
    else:
        raise ValueError(
            f"Multiple temperature values found in the state dict: {state_dict.keys()}. Please check the checkpoint."
        )


def build_global_embedding_network(
    ckpt_path: str | dict,
    adaptation_plan_path: str,
    txt_tf_embed_dim: int = 768,
    txt_tf_heads: int = 12,
) -> nn.Module:
    """
    Builds a classification network from a checkpoint path and an adaptation plan.
    Requires information about the text encoder embdeding dimension and its number of heads.

    Args:
        ckpt_path (str): Loaded checkpoint or Path to the checkpoint file with the weights.
        adaptation_plan_path (str): Path to the adaptation plan JSON file.
        txt_tf_embed_dim (int, optional): Embedding dimension of the text encoder. Defaults to 768.
        txt_tf_heads (int, optional): Embedding dimension of . Defaults to 12.

    Returns:
        nn.Module: _description_
    """
    state_dict = mb_load_ckpt(ckpt_path)

    network: AbstractDynamicNetworkArchitectures
    network = build_dense_embedding_network(ckpt_path, adaptation_plan_path)
    # We need replace the up_projection layer to match the expected input shape for our own classification stuff.
    # Before we do  "b (w h d) c -> b c w h d"
    network.up_projection = Rearrange("b c w h d -> b (w h d) c")

    proj_statedict_vvl_vlg = {
        k.replace("vision_language_projection.", ""): v
        for k, v in state_dict.items()
        if k.startswith("vision_language_projection.")
    }
    proj_statedict_vlm = {
        k.replace("vision_model.3.", ""): v for k, v in state_dict.items() if k.startswith("vision_model.3.")
    }
    if len(proj_statedict_vvl_vlg) > 0:
        embed_dim = proj_statedict_vvl_vlg["weight"].shape[1]
    else:
        embed_dim = proj_statedict_vlm["weight"].shape[1]
    projection_layer = nn.Linear(embed_dim, txt_tf_embed_dim)  # Always exists

    if len(proj_statedict_vvl_vlg) == 2:  # weight + Bias
        projection_layer.load_state_dict(proj_statedict_vvl_vlg)
    elif len(proj_statedict_vlm) == 2:
        projection_layer.load_state_dict(proj_statedict_vlm)
    else:
        raise ValueError("Unexpected state-dict structure")

    pooling_layer = get_vision_pooling_class_from_ckpt_path(ckpt_path, txt_tf_embed_dim, txt_tf_heads)
    pooling_state_dict = {
        k.replace("vision_pooling.", ""): v for k, v in state_dict.items() if k.startswith("vision_pooling.")
    }
    pooling_layer.load_state_dict(pooling_state_dict)
    joint_module = nn.Sequential(
        network,
        projection_layer,
        pooling_layer,
    ).to("cuda:0")

    return joint_module

