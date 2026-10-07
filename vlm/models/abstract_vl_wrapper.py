from __future__ import annotations
import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from peft import PeftModel
from transformers import PreTrainedModel

from vlm.configs.model.model_config import ModelConfig
from vlm.configs.model.trainable_params.trainable_linear_layer import TrainableLinearConfig
from vlm.configs.model.trainable_params.trainable_module import TrainableModuleConfig
from vlm.configs.run_config import RunConfig
from vlm.utils.model_utils import create_model_wrapper
from vlm.configs.compatibility import normalize_saved_config
from vlm.utils.training.checkpointing import resolve_ckpt_dir_in_run_dir
from vlm.utils.training.checkpoint_state import (
    initialize_checkpoint_state, load_model_checkpoint,
)
from vlm.utils import logger
from vlm.utils.class_utils import getattr_nested, setattr_nested
from vlm.utils.load_and_save_files import load_yaml


class AbstractVLWrapper(ABC):
    def __init__(self,
                 model_cfg: ModelConfig,
                 vision_input_type: str,
                 vision_embed_dim: int,
                 vision_embed_size: tuple[int, int, int],
                 text_only: bool = False,
                 max_response_tokens: int | None = None,
                 local_files_only=True,
                 device=None):
        # init in setup_model()
        self._model_str = None
        self.model: PreTrainedModel | None = None
        self.tokenizer = None
        self.processor = None
        self.downsample_factor = None

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_input_type = vision_input_type
        self.vision_embed_dim = vision_embed_dim
        self.vision_embed_size = vision_embed_size
        self.text_only = text_only
        self.max_response_tokens = max_response_tokens
        if max_response_tokens is not None and max_response_tokens <= 0:
            raise ValueError("'max_response_tokens' must be positive or None.")
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

    def initialize_model(self) -> PreTrainedModel:
        """Assemble and prepare the model before restoring checkpoint weights.

        PEFT layers must exist before checkpoint tracking and weight restoration.
        Checkpoint selection and restoration remain the caller's responsibility.
        """
        self.setup_model()
        self.configure_model_for_training()
        self.initialize_checkpoint_tracking()
        return self.model

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

    def _truncate_and_pad(self, batch, prompt_lengths: list[int]):
        """Limit each response independently, preserving its prompt, then right-pad.

        Shared here because some wrappers use a tokenizer directly rather than a processor.
        """
        if self.max_response_tokens is not None:
            for i, prompt_length in enumerate(prompt_lengths):
                max_length = prompt_length + self.max_response_tokens
                for sequences in batch.values():
                    sequences[i] = sequences[i][:max_length]
        return self.tokenizer.pad(batch, padding=True, padding_side="right", return_tensors="pt")

    @staticmethod
    def _replace_vision_content_type(messages: list[dict], content_type: str) -> list[dict]:
        """Replace dataset `vision` item types with the type expected by a processor.

        For example, native Qwen processors expect `video` instead of `vision`.
        Preserve item order and all other fields, including `vision_input_path`.
        """
        converted_messages = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = [
                    {**item, "type": content_type} if item.get("type") == "vision" else item
                    for item in content
                ]
            converted_messages.append({**message, "content": content})
        return converted_messages

    def _apply_chat_template(self, messages: list[dict], add_generation_prompt: bool) -> str:
        """Format structured messages using the model's processor."""
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _prepare_vision_inputs(
        self, vision_inputs: list
    ) -> tuple[torch.Tensor, list[int], list[torch.Tensor], dict]:
        """Convert a flat image list, prepare adapter metadata, and stack matching shapes.

        Callers handle model-specific validation and multi-image flattening.
        """
        vision_inputs = [
            vi.detach().cpu().numpy() if isinstance(vi, torch.Tensor) else np.asarray(vi)
            for vi in vision_inputs
        ]
        if not vision_inputs:
            raise ValueError(
                f"Received an empty vision input list in {type(self).__name__}.collate_fn"
            )

        num_vision_tokens, vision_position_ids, adapter_args =  self.vision_adapter.get_adapter_args(
            vision_inputs,
            self.vision_input_type,
            downsample_factor=self.downsample_factor,
        )
        return (
            torch.from_numpy(np.stack(vision_inputs)),
            num_vision_tokens,
            vision_position_ids,
            adapter_args,
        )

    def get_model_specific_training_args(self):
        """Return default precision settings; subclasses can override these as needed."""
        return {"bf16": True, "tf32": True}

    @property
    def module_attr_map(self) -> dict[str, str]:
        """
        Map component names to attribute paths relative to `self.model`.

        Subclasses with a different module structure should override this mapping.
        """
        return {
            "vision_encoder": "model.vision_encoder",
            "adapter": "model.vision_adapter",
            "language_model": "model.language_model",
            "lm_head": "lm_head",
        }

    @property
    def vision_adapter(self) -> torch.nn.Module | None:
        """Return the current adapter, including any PEFT wrapper."""
        if self.model is None:
            return None
        return getattr_nested(self.model, self.module_attr_map["adapter"])

    def _setup_vision_components(self) -> None:
        """Build and attach vision components after loading the pretrained backbone.

        Components initialize their own weights; no global HF initialization follows.
        Subclasses set `downsample_factor` according to their encoder/config convention.
        """
        if self.model_cfg.vision_encoder is not None:
            encoder = self.model_cfg.build_vision_encoder()
            setattr_nested(self.model, self.module_attr_map["vision_encoder"], encoder)
        if self.model_cfg.adapter is not None:
            language_model = getattr_nested(self.model, self.module_attr_map["language_model"])
            text_embed_dim = language_model.get_input_embeddings().weight.shape[-1]
            adapter = self.model_cfg.build_adapter(
                vision_embed_dim=self.vision_embed_dim,
                vision_embed_size=self.vision_embed_size,
                text_embed_dim=text_embed_dim,
            )
            setattr_nested(self.model, self.module_attr_map["adapter"], adapter)

    def configure_model_for_training(self) -> None:
        """
        Apply the configured training mode to each model component.

        The `module_attr_map` property specifies the component paths within the model.
        Components can be frozen, fully trainable, or wrapped with PEFT layers. A component may also
        specify a PEFT configuration together with ``mode='frozen'``. In that case, the PEFT wrapper is
        constructed so weights from an earlier partial checkpoint have matching parameter names, and
        the entire wrapped component is then frozen.
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
            if (mode == "frozen" or t_o_enabled_and_not_lm) and trainable_component_cfg.peft is None:
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
            elif mode == "peft" or trainable_component_cfg.peft is not None:
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

                if mode == "frozen" or t_o_enabled_and_not_lm:
                    wrapped.requires_grad_(False)
                logger.info_main(f"  > {module_name}: PEFT ({trainable_component_cfg.peft.type.upper()})")
            else:
                raise ValueError(f"Unknown mode in {TrainableModuleConfig.__name__}: {mode}")

    def initialize_checkpoint_tracking(self) -> None:
        """Register state that must be included in partial model checkpoints.

        This must run after ``configure_model_for_training`` so any PEFT parameters already exist.
        It registers currently trainable parameters and persistent buffers. Registered keys remain
        required if their parameters are frozen later or their weights were loaded from a previous
        partial checkpoint.
        """

        initialize_checkpoint_state(self.model)

    @staticmethod
    def load_from_run_dir(run_dir: str | Path, mode: Literal["best", "final"] = "final") -> tuple[AbstractVLWrapper, RunConfig, Path]:
        """
        Load a trained VLM model from a run directory.

        Args:
            run_dir: Path to the run directory containing checkpoint subfolders and the final configuration file.
            mode: Which checkpoint to restore.
                - "best": Load the checkpoint selected by 'trainer.metric_for_best_model'.
                - "final": Load the last saved checkpoint after training.

        Returns:
            An instance of the model with weights restored.
        """
        ckpt_dir = resolve_ckpt_dir_in_run_dir(run_dir=run_dir, mode=mode)
        logger.info_main(f"Loading {mode} model from {str(run_dir)}")

        return AbstractVLWrapper.load_from_checkpoint_dir(ckpt_dir) + (ckpt_dir,)

    def load_weights_from_checkpoint_dir(
        self,
        ckpt_dir: str | Path,
    ) -> tuple[list[str], list[str]]:
        """
        Load only model weights into an already constructed model.
        Assumes self.model exists and trainable wrappers (e.g. PEFT) are already applied.
        """
        if self.model is None:
            raise RuntimeError("self.model is None. Call setup_model() first.")

        return load_model_checkpoint(self.model, ckpt_dir)

    @staticmethod
    def load_from_checkpoint_dir(ckpt_dir: str | Path) -> tuple[AbstractVLWrapper, RunConfig]:
        """
        Restore a VLM model from a checkpoint directory.

        Args:
            ckpt_dir: Path to the checkpoint directory.
                Must contain 'model.safetensors'. Architecture configuration comes
                from the parent run directory's 'final_config.yml'.

        Returns:
            An instance of the model with weights restored.
        """
        ckpt_dir = Path(ckpt_dir)
        cfg_dict = load_yaml(ckpt_dir.parent / "final_config.yml")["run"]
        cfg_dict, config_warnings = normalize_saved_config(cfg_dict)
        for message in config_warnings:
            logger.warning_main(message)
        cfg = RunConfig(**cfg_dict)

        logger.info_main(f"Loading model from {str(ckpt_dir)}")

        vlm_model = create_model_wrapper(cfg)
        vlm_model.initialize_model()
        vlm_model.load_weights_from_checkpoint_dir(ckpt_dir)
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
