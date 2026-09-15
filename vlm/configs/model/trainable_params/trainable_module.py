from abc import ABC, abstractmethod
from typing import Literal, Annotated, Union
from peft import LoraConfig
from pydantic import Field, model_validator
from vlm.configs.base import DiscriminatedConfigBase, ConfigBase


class DiscriminatedPEFTConfigBase(DiscriminatedConfigBase, ABC):
    @abstractmethod
    def to_peft_config(self, *args, **kwargs):
        """Return the PEFT config"""
        raise NotImplementedError


class PEFTLoraConfig(DiscriminatedPEFTConfigBase):
    type: Literal["lora"]
    r: int = Field(8, description="LoRA rank.")
    lora_alpha: int = Field(16, description="Scaling factor applied to LoRA updates.")
    lora_dropout: float = Field(0.05, description="Dropout probability applied to LoRA branch.")
    target_modules: list[str] | str = Field(..., description="Name(s) of module(s) where LoRA adapters are injected.")
    modules_to_save: list[str] | None = Field(None, description="Extra modules to keep trainable and save with adapter.")
    bias: Literal["none", "lora_only", "all"] = Field("none", description="Bias handling.")

    def to_peft_config(self):
        return LoraConfig(**self.to_dict_no_type())


# Discriminated union for different PEFT configs
PEFTConfig = Annotated[Union[PEFTLoraConfig], Field(discriminator="type")]

class TrainableModuleConfig(ConfigBase):
    mode: Literal["frozen", "full", "peft"] = Field(...,
        description="Training mode: freeze module ('frozen'), finetune fully ('full'), or apply PEFT ('peft')."
    )
    peft: PEFTConfig | None = Field(
        None,
        description="PEFT configuration (e.g. LoRA, LoHa); required if mode='peft'."
    )

    @model_validator(mode="after")
    def check_peft_consistency(self):
        if self.mode == "peft" and self.peft is None:
            raise ValueError("Must provide 'peft' config when mode='peft'.")
        if self.mode == "full" and self.peft is not None:
            raise ValueError("'peft' must be None for mode='full'; frozen PEFT modules are supported.")
        return self
