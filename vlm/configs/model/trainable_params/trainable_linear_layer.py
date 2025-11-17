from abc import ABC, abstractmethod
from typing import Literal, Annotated, Union
from pydantic import Field, model_validator
from vlm.configs.base import DiscriminatedConfigBase, ConfigBase
from peft.tuners.lora import Linear as LoraLinear
import torch.nn as nn


class DiscriminatedLinearBase(DiscriminatedConfigBase, ABC):
    @abstractmethod
    def to_peft_linear_layer(self, *args, **kwargs):
        """Return the PEFT config"""
        raise NotImplementedError


class PEFTLoraLinearConfig(DiscriminatedLinearBase):
    type: Literal["lora"]
    r: int = Field(8, description="LoRA rank.")
    lora_alpha: int = Field(16, description="Scaling factor applied to LoRA updates.")
    lora_dropout: float = Field(0.05, description="Dropout probability applied to LoRA branch.")

    def to_peft_linear_layer(self, base_linear_layer: nn.Module):
        return LoraLinear(
            base_layer=base_linear_layer,
            adapter_name="default",
            **self.to_dict_no_type(),
        )


# Discriminated union for different PEFT Linear layers configs
PEFTLinearConfig = Annotated[Union[PEFTLoraLinearConfig], Field(discriminator="type")]

class TrainableLinearConfig(ConfigBase):
    mode: Literal["frozen", "full", "peft"] = Field(...,
        description="Training mode: freeze module ('frozen'), finetune fully ('full'), "
                    "or convert to PEFT Linear layer ('peft')."
    )
    peft: PEFTLinearConfig | None = Field(
        None,
        description="PEFT Linear layer; required if mode='peft'."
    )

    @model_validator(mode="after")
    def check_peft_consistency(self):
        if self.mode == "peft" and self.peft is None:
            raise ValueError("Must provide 'peft' config when mode='peft'.")
        if self.mode != "peft" and self.peft is not None:
            raise ValueError("'peft' must be None unless mode='peft'.")
        return self
