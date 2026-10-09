"""Portable config paths with explicit environment-variable expansion."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_core import core_schema

_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class ConfigPath:
    """Keep the source expression for saving; expand it only when requested."""

    expression: str
    _runtime_path: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.expression = str(self.expression)

    def __str__(self) -> str:
        return self.expression

    def get_resolved_path(self, *, field: str = "path") -> str:
        """Expand ${NAME} and ~, or return the runtime override when set."""
        if self._runtime_path is not None:
            return self._runtime_path
        raw = self.expression
        if not raw:
            raise ValueError(f"{field}: path must not be empty")
        if "${" in _VARIABLE.sub("", raw):
            raise ValueError(f"{field}: invalid path expression {raw!r}; use ${{NAME}}")

        def replace(match: re.Match) -> str:
            name = match.group(1)
            value = os.environ.get(name)
            if not value:
                raise ValueError(
                    f"{field}: environment variable {name!r} is unset or empty in {raw!r}"
                )
            return value

        return os.path.expanduser(_VARIABLE.sub(replace, raw))

    def set_runtime_path(self, value: str | Path) -> None:
        """Use a temporary location without changing the saved expression."""
        self._runtime_path = str(value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Load strings as ConfigPath objects and dump their source expressions."""
        from_string = core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema(min_length=1)
        )
        return core_schema.json_or_python_schema(
            json_schema=from_string,
            python_schema=core_schema.union_schema(
                [core_schema.is_instance_schema(cls), from_string]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda path: path.expression, return_schema=core_schema.str_schema()
            ),
        )
