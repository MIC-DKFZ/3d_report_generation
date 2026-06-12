from __future__ import annotations
import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal
import safetensors.torch
import torch
from peft import PeftModel
from transformers import PreTrainedModel
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from vlm.configs.model.model_config import ModelConfig
from vlm.configs.model.trainable_params.trainable_linear_layer import TrainableLinearConfig
from vlm.configs.model.trainable_params.trainable_module import TrainableModuleConfig
from vlm.configs.run_config import RunConfig
from vlm.utils import logger
from vlm.utils.helper import getattr_nested, setattr_nested, resolve_ckpt_dir_in_run_dir
from vlm.utils.load_and_save_files import load_yaml
from vlm.utils.load_sharded_checkpoint import load_sharded_checkpoint, apply_regex_subs


class AbstractVLMModel(ABC):
    def __init__(self,
                 model_cfg: ModelConfig,
                 vision_input_type: str,
                 vision_embed_dim: int,
                 vision_embed_size: tuple[int, int, int],
                 text_only: bool = False,
                 do_truncation: bool = False,
                 truncate_after: str | None = None,
                 truncate_at: int | None = None,
                 local_files_only=True,
                 device=None):
        # init in setup_model()
        self._model_str = None
        self.model: PreTrainedModel | None = None
        self.tokenizer = None
        self.processor = None
        self.downsample_factor = None
        self.vision_adapter = None

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_input_type = vision_input_type
        self.vision_embed_dim = vision_embed_dim
        self.vision_embed_size = vision_embed_size
        self.text_only = text_only
        self.do_truncation = do_truncation
        self.truncate_after = truncate_after
        self.truncate_at = truncate_at
        self.local_files_only = local_files_only

        self.model_cfg = model_cfg

        self.forbidden_name_patterns_for_wd = [
            r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)",
        ]

    @abstractmethod
    def setup_model(self, **kwargs):
        """
        Initialize the model, tokenizer, and processor.
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

    def apply_trainable_config(self) -> None:
        """
        Apply the TrainableParamsConfig in self.trainable_params to each submodule.

        The 'self.module_attr_map' property must be defined in the subclass before calling this function.
        """
        logger.info_main("[Trainable Parameters] Setting trainable parameters and/or adding PEFT layers according to input config:")
        for module_name, module_attr_path in self.module_attr_map.items():
            trainable_component_cfg: TrainableModuleConfig | TrainableLinearConfig = getattr(
                self.model_cfg.trainable_params, module_name)

            # trainable_component_cfg can be None (e.g., when embeddings are used directly,
            # meaning no vision encoder and thus no TrainableModuleConfig for it)
            if trainable_component_cfg is None:
                logger.info_main(f"  > {module_name}: NON-EXISTENT")
                continue

            # if text_only is enabled, freeze vision encoder and adapter
            t_o_enabled_and_not_lm = module_name in ("vision_encoder", "adapter") and self.text_only

            mode = trainable_component_cfg.mode
            if mode == "frozen" or t_o_enabled_and_not_lm:
                module: torch.nn.Module = getattr_nested(self.model, module_attr_path)
                for name, param in module.named_parameters():
                    param.requires_grad = False
                text_only_suffix = "" if not t_o_enabled_and_not_lm else " (TEXT-ONLY INPUT)"
                logger.info_main(f"  > {module_name}: frozen" + text_only_suffix)
            elif mode == "full":
                module: torch.nn.Module = getattr_nested(self.model, module_attr_path)
                for name, param in module.named_parameters():
                    param.requires_grad = True
                logger.info_main(f"  > {module_name}: full finetuning")
            elif mode == "peft":
                original = getattr_nested(self.model, module_attr_path)

                # PEFT injects adapters (e.g. LoRA) into specific submodules of a larger nn.Module.
                # For composite modules, use the PeftModel class. For single layers (e.g. 'lm_head'),
                # replace the layer directly with PEFT's Linear class (e.g. peft.tuners.lora.Linear for LoRA).
                if isinstance(trainable_component_cfg, TrainableModuleConfig):
                    peft_config = trainable_component_cfg.peft.to_peft_config()
                    wrapped = PeftModel(original, peft_config)
                elif isinstance(trainable_component_cfg, TrainableLinearConfig):
                    base_linear_layer = original
                    wrapped = trainable_component_cfg.peft.to_peft_linear_layer(base_linear_layer=base_linear_layer)

                    # Some models (e.g. Qwen2.5-VL) tie 'lm_head.weight' to 'embed_tokens.weight'.
                    # After wrapping, the tied weight lives under 'lm_head.base_layer.weight', so update
                    # '_tied_weights_keys' to keep Transformers’ save/load logic consistent.
                    if (module_name == "lm_head"
                            and getattr(self.model.config.get_text_config(decoder=True), "tie_word_embeddings", True)):
                        self.model._tied_weights_keys = ["lm_head.base_layer.weight"]
                        logger.info_main("    > Updated '_tied_weights_keys' from ['lm_head.weight'] to ['lm_head.base_layer.weight']")
                else:
                    raise ValueError(f"Unknown class: {type(trainable_component_cfg).__name__}")

                setattr_nested(self.model, module_attr_path, wrapped)
                logger.info_main(f"  > {module_name}: PEFT ({trainable_component_cfg.peft.type.upper()})")
            else:
                raise ValueError(f"Unknown mode in {TrainableModuleConfig.__name__}: {mode}")

    @staticmethod
    def load_from_run_dir(run_dir: str | Path, mode: Literal["best_metric", "best", "final"] = "final") -> tuple[AbstractVLMModel, RunConfig, Path]:
        """
        Load a trained VLM model from a run directory.

        Args:
            run_dir: Path to the run directory containing checkpoint subfolders and the final configuration file.
            mode: Which checkpoint to restore.
                - "best_metric": Load the checkpoint with the custom score (located in 'best_model-*').
                - "best": Load the checkpoint with the best validation score.
                - "final": Load the last saved checkpoint after training.

        Returns:
            An instance of the model with weights restored.
        """
        ckpt_dir = resolve_ckpt_dir_in_run_dir(run_dir=run_dir, mode=mode)
        logger.info_main(f"Loading {mode} model from {str(run_dir)}")

        return AbstractVLMModel.load_from_checkpoint_dir(ckpt_dir) + (ckpt_dir,)

    def load_weights_from_checkpoint_dir(
        self,
        ckpt_dir: str | Path,
        allow_unexpected_keys: bool = False,
    ) -> tuple[list[str], list[str]]:
        """
        Load only model weights into an already constructed model.
        Assumes self.model exists and trainable wrappers (e.g. PEFT) are already applied.
        """
        if self.model is None:
            raise RuntimeError("self.model is None. Call setup_model() first.")

        ckpt_dir = Path(ckpt_dir)
        single_path = ckpt_dir / SAFE_WEIGHTS_NAME
        index_path = ckpt_dir / SAFE_WEIGHTS_INDEX_NAME
        checkpoint_conversion_mapping = getattr(self.model, "_checkpoint_conversion_mapping", {})

        if single_path.exists():
            state_dict = safetensors.torch.load_file(single_path)
            state_dict = apply_regex_subs(state_dict, checkpoint_conversion_mapping)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            if unexpected_keys and not allow_unexpected_keys:
                raise KeyError(f"Unexpected keys in loaded checkpoint: {unexpected_keys}")

            return missing_keys, unexpected_keys

        if index_path.exists():
            load_sharded_checkpoint(
                model=self.model,
                folder=ckpt_dir,
                allow_unexpected_keys=allow_unexpected_keys,
                key_mapping=checkpoint_conversion_mapping
            )
            return [], []

        raise FileNotFoundError(
            f"Neither {SAFE_WEIGHTS_NAME!r} nor {SAFE_WEIGHTS_INDEX_NAME!r} found in {ckpt_dir}"
        )

    @staticmethod
    def load_from_checkpoint_dir(ckpt_dir: str | Path, allow_unexpected_keys=False) -> tuple[AbstractVLMModel, RunConfig]:
        """
        Restore a VLM model from a checkpoint directory.

        Args:
            ckpt_dir: Path to the checkpoint directory.
                Must contain either a single 'model.safetensors' file or a sharded
                'model.safetensors.index.json' with corresponding shard files.
            allow_unexpected_keys: If 'False', strictly enforce that all keys in the checkpoint
                exist in the model’s architecture. If 'True', allow unexpected keys.

        Returns:
            An instance of the model with weights restored.
        """
        ckpt_dir = Path(ckpt_dir)
        cfg_dict = load_yaml(ckpt_dir.parent / "final_config.yml")["run"]
        cfg = RunConfig(**cfg_dict)

        logger.info_main(f"Loading model from {str(ckpt_dir)}")

        from vlm.registries.vlm_registry import VLM_REGISTRY
        vlm_model_class = VLM_REGISTRY[cfg.model.model_type].model_class
        vlm_model: AbstractVLMModel = vlm_model_class(
            model_cfg=cfg.model,
            vision_input_type=cfg.data.vision_input_type,
            vision_embed_dim=cfg.get_vision_embed_dim(),
            vision_embed_size=cfg.get_vision_embed_size(),
            text_only=cfg.data.text_only,
            do_truncation=cfg.data.do_truncation,
            truncate_after=cfg.data.truncate_after,
            truncate_at=cfg.data.truncate_at,
            local_files_only=cfg.general.local_files_only
        )
        vlm_model.setup_model()
        vlm_model.apply_trainable_config()
        vlm_model.load_weights_from_checkpoint_dir(
            ckpt_dir=ckpt_dir,
            allow_unexpected_keys=allow_unexpected_keys
        )
        return vlm_model, cfg

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
        logger.info_main(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info_main(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
