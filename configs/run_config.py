from configs.base import ConfigBase
from configs.data_config import DataConfig
from configs.general_config import GeneralConfig
from configs.model.model_config import ModelConfig
from configs.trainer_config import SFTTrainerConfig


class RunConfig(ConfigBase):
    """
    Aggregates all sub-configurations for a training run.

    Currently, the following configuration are held by the RunConfig class:
            - general: general info and configurations
            - model: model configs, trainable parameters
            - data: dataset, images/embeddings, data augs, number of samples
            - trainer: SFTTrainer hyperparameters
    """
    general: GeneralConfig
    model: ModelConfig
    data: DataConfig
    trainer: SFTTrainerConfig