import gc
from abc import ABC, abstractmethod
import torch
from peft import get_peft_model

from configs.model.model_config import ModelConfig
from configs.model.trainable_params_config import TrainableParamsConfig, TrainableModuleConfig
from utils import logger
from utils.helper import getattr_nested, setattr_nested


class AbstractVLMModel(ABC):
    def __init__(self, model_cfg: ModelConfig, vision_input_type: str, checkpoint_path = None, local_files_only=True, device=None):
        self.model_id = model_cfg.model_id
        self.vision_encoder_path = model_cfg.vision_encoder.vision_encoder_path
        self.vision_input_type = vision_input_type
        self.checkpoint_dir = checkpoint_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.local_files_only = local_files_only

        self.trainable_params: TrainableParamsConfig = model_cfg.trainable_params

    @abstractmethod
    def setup_model(self, bnb_config=None):
        """
        Initialize the model, tokenizer, and processor.

        Args:
            bnb_config: Optional BitsAndBytesConfig for quantization
        """
        pass

    @abstractmethod
    def collate_fn(self, batch):
        """
        Custom collate function for the model's data processing.

        Args:
            batch: A batch of data samples

        Returns:
            A dictionary with model inputs and targets
        """
        pass

    @abstractmethod
    def get_model_specific_training_args(self):
        """
        Get model-specific training arguments.

        Returns:
            A dictionary with model-specific training arguments
        """
        pass

    @property
    @abstractmethod
    def module_attr_map(self) -> dict[str, str]:
        """
        Mapping of submodule names to their attribute paths (relative to 'self.model'!!!).

        Required keys:
          - "vision_encoder"
          - "adapter"
          - "language_model"
          - "lm_head"

        Modules might be nested within the model, e.g., 'self.model.model.vision_encoder'.
        Each value should be a string path.

        Example
        -------
        {
            "vision_encoder": "model.vision_encoder",
            "adapter": "model.vision_adapter",
            "language_model": "model.language_model",
            "lm_head": "lm_head",
        }
        """
        ...

    def set_trainable_params(self) -> None:
        """
        Apply the TrainableParamsConfig in self.trainable_params to each submodule.

        The 'self.module_attr_map' property must be defined in the subclass before calling this function.
        """
        logger.info("Setting trainable parameters according to input config:")
        for module_name, module_attr_path in self.module_attr_map.items():
            trainable_module_cfg: TrainableModuleConfig = getattr(self.trainable_params, module_name)
            mode = trainable_module_cfg.mode
            if mode == "frozen":
                module: torch.nn.Module = getattr_nested(self.model, module_attr_path)
                for name, param in module.named_parameters():
                    param.requires_grad = False
                logger.info(f"  > {module_name}: frozen")
            elif mode == "full":
                module: torch.nn.Module = getattr_nested(self.model, module_attr_path)
                for name, param in module.named_parameters():
                    param.requires_grad = True
                logger.info(f"  > {module_name}: full finetuning")
            elif mode == "peft":
                peft_config = trainable_module_cfg.peft.to_peft_config()
                original_module = getattr_nested(self.model, module_attr_path)
                wrapped_module = get_peft_model(original_module, peft_config)
                setattr_nested(self.model, module_attr_path, wrapped_module)
                logger.info(f"  > {module_name}: PEFT ({trainable_module_cfg.peft.type.upper()})")
            else:
                raise ValueError(f"Unknown mode in {TrainableModuleConfig.__name__}: {mode}")

    def clear_memory(self):
        """Clear GPU memory and cached objects"""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
