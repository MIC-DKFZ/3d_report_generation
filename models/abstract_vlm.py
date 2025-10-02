import gc
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from utils import logger


class AbstractVLMModel(ABC):
    def __init__(self, model_id, train_only_vision_adapter: bool, train_vision_encoder=False, vision_encoder_path=None, device=None, local_files_only=True):
        self.model_id = model_id
        # TODO: Make a default argument since its not needed for inference
        self.train_only_vision_adapter = train_only_vision_adapter
        self.train_vision_encoder = train_vision_encoder
        self.vision_encoder_path = vision_encoder_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.local_files_only = local_files_only

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
    def get_trainable_params(self):
        """
        Define which parameters should be trainable for this model.

        Returns:
            A dictionary with configuration for trainable parameters
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
