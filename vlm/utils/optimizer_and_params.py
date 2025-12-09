import torch.nn as nn
from transformers.trainer_pt_utils import get_parameter_names
from vlm.models.owl.modeling_hyper_qwen2 import Qwen2RMSNorm


def get_decay_parameter_names(model: nn.Module, forbidden_name_patterns_for_wd: list[str]) -> list[str]:
    decay_parameters = get_parameter_names(model, [nn.LayerNorm, Qwen2RMSNorm], forbidden_name_patterns_for_wd)
    return decay_parameters


def get_param_groups(
        lr_param_groups: list[tuple[str, str, float]],
        weight_decay: float,
        model: nn.Module,
        forbidden_name_patterns_for_wd: list[str],
) -> list[dict]:
    """
    Creates parameter groups separated by weight-decay requirements and optional module-specific learning rates.

    Args:
        lr_param_groups (list[tuple[str, str, float]]):
            A list of (module_name, module_attr_path, learning_rate) tuples, e.g.,
            [('vision_encoder', 'model.vision_model', 1e-4), ('vision_language_projector', 'model.projector', 1e-3)].
            For each tuple, parameters whose names start with the given module_attr_path
            are grouped together with the specified learning rate and appropriate
            weight-decay settings.
            To set a global learning rate for all trainable parameters,
            use [("global", "", lr)], as str.startswith("") always return True.
        weight_decay (float): The weight decay value to be applied to affected parameters.
        model (nn.Module): The model to train.
        forbidden_name_patterns_for_wd (list[str]): Layer-name patterns for which weight decay should not be applied.

    Returns:
        list[dict]: Parameter groups
    """

    decay_param_names = set(get_decay_parameter_names(model, forbidden_name_patterns_for_wd))

    param_groups = []
    assigned_param_ids = set()

    for module_name, module_attr_path, learning_rate in lr_param_groups:
        group_decay = []
        group_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in assigned_param_ids:
                continue
            if name.startswith(module_attr_path):
                if name in decay_param_names:
                    group_decay.append(param)
                else:
                    group_no_decay.append(param)
                assigned_param_ids.add(id(param))

        if group_decay:
            param_groups.append({
                "name": f"{module_name}_DECAY",
                "params": group_decay,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            })

        if group_no_decay:
            param_groups.append({
                "name": f"{module_name}_NO_DECAY",
                "params": group_no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
            })

    # if using 'lr_param_groups', search for unused parameters
    if lr_param_groups[0][0] != "":
        missing_params = [
            n for n, p in model.named_parameters()
            if p.requires_grad and id(p) not in assigned_param_ids
        ]
        if missing_params:
            raise ValueError(f"{len(missing_params)} trainable parameters were NOT assigned to the optimizer:\n"
                                f"{missing_params}")

    return param_groups

