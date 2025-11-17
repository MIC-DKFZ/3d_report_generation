from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from vlm.utils.find_class_by_name import recursive_find_python_class
import vlm

class ConfigBase(BaseModel):
    """
    Common base class for all config models.

    - Provides strict validation by forbidding unknown fields.
    - Serves as the root for all configuration schemas.
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )


class DiscriminatedConfigBase(ConfigBase):
    """
    Base class for configs used in discriminated unions.

    - Requires a 'type' field for discrimination.
    - Subclasses must override 'type' with a Literal value.
    """
    type: str

    def to_dict_no_type(self):
        return self.model_dump(exclude={"type"})


class DynamicClassConfigBase(ConfigBase):
    """
    Base class for configs that only hold the specific class name ('class_name') that is to be
    instantiated and returned via 'to_instance()' and its arguments ('class_args').
    """
    target_module = None  # Overwrite in subclass

    class_name: str = Field(..., description="Name of the class to load")
    class_kwargs: dict = Field(..., description="Keyword arguments to instantiate the class with")

    _resolved_class: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def resolve_target(self):
        """Resolve and store the Python object after validation."""
        self._resolved_class = recursive_find_python_class(
            target_module=str(Path(vlm.__path__[0], *self.target_module.split("."))),
            class_name=self.class_name,
            current_module=self.target_module
        )
        if self._resolved_class is None:
            raise ValueError(f"Could not find {self.class_name:!r} in {vlm.__path__[0]:!r}")
        return self

    def to_instance(self) -> Any:
        return self._resolved_target(**self.class_kwargs)