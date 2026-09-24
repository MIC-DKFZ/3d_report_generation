from abc import ABC, abstractmethod
from pathlib import Path
from types import UnionType
from typing import Any, ClassVar, Callable, Type, OrderedDict, List, Tuple, Union, get_type_hints, get_origin
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter, model_validator
from vlm.utils.find_class_by_name import recursive_find_python_class
import vlm
import inspect
from inspect import Parameter

from vlm.utils.helper import get_class_from_registry


def _get_signatures(cls: Type) -> OrderedDict[str, Any]:
    """
    Collects and merges the *type hints* of all __init__ arguments
    from a class and its ancestors in the MRO.
    Returns: OrderedDict[str, Any] where Any is the type hint (e.g., int, list[str]).
    """
    mro_chain = inspect.getmro(cls)
    classes_to_check = mro_chain[:-1]  # Exclude 'object'

    all_hints: OrderedDict[str, Any] = OrderedDict()

    for current_cls in classes_to_check:
        if '__init__' not in current_cls.__dict__:
            continue

        init_method: Callable = current_cls.__init__

        # Get type hints
        try:
            hints = get_type_hints(init_method)
        except Exception:
            hints = {}

        # 1. Get parameter names from the signature
        try:
            sig = inspect.signature(init_method)
        except ValueError:
            continue

        for name in sig.parameters:
            if name in ('self', 'args', 'kwargs'):
                continue

            # 2. Merge: child class hints take precedence
            if name not in all_hints:
                # Add the hint if it exists, otherwise use 'Any' or similar placeholder
                all_hints[name] = hints.get(name, Parameter.empty)

    return all_hints


def _convert_value(value: Any, expected_type: Any) -> Any:
    """
    Attempts to convert a value (typically a string from argparse)
    to the expected type from the type hint.
    """
    if expected_type in (Parameter.empty, Any):
        return value

    origin = get_origin(expected_type)
    if expected_type is bool or origin in (Union, UnionType, list, List, tuple, Tuple):
        return TypeAdapter(
            expected_type, config=ConfigDict(arbitrary_types_allowed=True)
        ).validate_python(value)

    if expected_type in (list, tuple):
        return value  # Avoid splitting strings with list(value) or tuple(value).

    if not isinstance(expected_type, type):
        return value  # Leave unsupported annotations unchanged.

    try:
        return expected_type(value)
    except (ValueError, TypeError) as exc:
        if isinstance(value, expected_type):
            return value
        raise TypeError(
            f"Could not convert value '{value}' to expected type '{expected_type}'"
        ) from exc


class ConfigBase(BaseModel, ABC):
    """
    Common base class for all config models.

    - Provides strict validation by forbidding unknown fields.
    - Serves as the root for all configuration schemas.
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=False,
        use_enum_values=True
    )


class DiscriminatedConfigBase(ConfigBase, ABC):
    """
    Base class for configs used in discriminated unions.

    - Requires a 'type' field for discrimination.
    - Subclasses must override 'type' with a Literal value.
    """
    type: str

    def to_dict_no_type(self):
        return self.model_dump(exclude={"type"})


class DynamicClassConfigBase(ConfigBase, ABC):
    """
    Base class for configs that only hold the specific class name ('class_name') that is to be
    instantiated and returned via 'to_instance()' and its arguments ('class_args').
    """
    class_name: str = Field(..., description="Name of the class to load.")
    class_kwargs: dict = Field({}, description="Keyword arguments to instantiate the class with.")

    _resolved_class: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _resolve_target(self):
        self.resolve_target()
        self._sanitize_class_kwargs(_get_signatures(self._resolved_class))
        return self

    @abstractmethod
    def resolve_target(self):
        """Resolve and store the Python object after validation."""
        ...

    def _sanitize_class_kwargs(self, signatures: OrderedDict[str, Any]):
        """
        Checks and converts argument values in self.class_kwargs based on
        the type hints provided in the signatures.
        """
        for key, value in self.class_kwargs.items():
            if key in signatures:
                self.class_kwargs[key] = _convert_value(value, signatures[key])

    def to_instance(self, **kwargs) -> Any:
        return self._resolved_class(**kwargs, **self.class_kwargs)


class RecursiveFindClassConfigBase(DynamicClassConfigBase):
    """
    This class resolves the target by recursively searching modules and subpackages
    starting at 'folder_to_search' to find a class by name.
    """
    folder_to_search: ClassVar[str]  # Overwrite in subclass;
                                     # Should point to the module where to start the recursive search
                                     # for the target class. Example: 'vlm.adapters'

    def resolve_target(self):
        """Resolve and store the Python object after validation."""
        self._resolved_class = recursive_find_python_class(
            folder=str(Path(vlm.__path__[0], *self.folder_to_search.split(".")[1:])),
            class_name=self.class_name,
            current_module=self.folder_to_search
        )
        if self._resolved_class is None:
            raise ValueError(f"Could not find {self.class_name!r} in {vlm.__path__[0]!r}")
        return self

    def to_instance(self, **kwargs) -> Any:
        return self._resolved_class(**kwargs, **self.class_kwargs)


class RegistryClassConfigBase(DynamicClassConfigBase):
    """
    This class resolves the target by looking it up in registry list.
    """
    registry: ClassVar[list[type]]  # Overwrite in subclass;
                                    # Should point to the registry/list where the class is held

    def resolve_target(self):
        """Resolve and store the Python object after validation."""
        target_class = get_class_from_registry(self.class_name, self.registry)
        if target_class is None:
            raise ValueError(f"Could not find {self.class_name:!r} in {self.registry}")
        self._resolved_class = target_class
