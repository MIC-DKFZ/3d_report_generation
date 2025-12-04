import torch.nn as nn
from torch.optim import Optimizer
from transformers.trainer_pt_utils import get_parameter_names


def get_decay_parameter_names(model: nn.Module, forbidden_name_patterns_for_wd: list[str]) -> list[str]:
    decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns_for_wd)
    return decay_parameters


def create_optimizer_with_grouped_params(
        optimizer_class: type[Optimizer],
        optimizer_kwargs: dict,
        model: nn.Module,
        forbidden_name_patterns_for_wd: list[str]
):
    decay_parameters = get_decay_parameter_names(model, forbidden_name_patterns_for_wd)
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": optimizer_kwargs.get("weight_decay", 0.0),
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_class(optimizer_grouped_parameters, **optimizer_kwargs)

