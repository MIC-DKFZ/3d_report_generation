from vlm.adapters.base import AdapterBase
from vlm.configs.base import RecursiveFindClassConfigBase
from math import prod

class AdapterClassConfig(RecursiveFindClassConfigBase):
    folder_to_search = "vlm.adapters"

    def to_instance(
            self,
            vision_embed_dim: int,
            vision_embed_size: tuple[int, int, int],
            text_embed_dim: int
    ) -> AdapterBase:
        """
        Instantiate the resolved adapter class, evaluating any string expressions
        in `class_kwargs` using the provided embedding parameters.
        """

        # These are always given by default and thus should not be provided in the config
        default_adapter_kwargs = {
            "vision_embed_dim": vision_embed_dim,
            "text_embed_dim": text_embed_dim,
            "num_vision_tokens": prod(vision_embed_size)
        }
        evaluated_adapter_cfg_kwargs = {}
        for key, val in self.class_kwargs.items():
            evaluated_val = val
            if isinstance(val, str):
                evaluated_val = eval(val, {"__builtins__": None}, default_adapter_kwargs)
            evaluated_adapter_cfg_kwargs[key] = evaluated_val

        return self._resolved_class(**default_adapter_kwargs, **evaluated_adapter_cfg_kwargs)