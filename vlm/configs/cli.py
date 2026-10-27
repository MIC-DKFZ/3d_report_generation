import argparse
import json
from collections.abc import Container
from typing import Any

from pydantic import BaseModel


def add_config_cli_args(
    parser: argparse.ArgumentParser,
    default_model: BaseModel,
    parent_key: str = "",
) -> set[str]:
    """Register dotted CLI options and return paths accepting additional class kwargs."""
    from vlm.configs.base import DynamicClassConfigBase

    dynamic_kwargs_paths: set[str] = set()

    def register_fields(model: BaseModel | dict[str, Any], prefix: str) -> None:
        if isinstance(model, BaseModel):
            fields = (
                (key, field.annotation, getattr(model, key))
                for key, field in type(model).model_fields.items()
            )
        else:
            fields = ((key, type(value), value) for key, value in model.items())

        for key, annotation, value in fields:
            path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, BaseModel):
                register_fields(value, path)
                continue
            if key == "class_kwargs" and isinstance(model, DynamicClassConfigBase):
                assert isinstance(value, dict)
                dynamic_kwargs_paths.add(path)
                register_fields(value, path)
                continue

            if annotation == bool:
                if value:
                    parser.add_argument(
                        f"--{prefix}.no-{key}" if prefix else f"--no-{key}",
                        action="store_false",
                        dest=path,
                        default=argparse.SUPPRESS,
                        help=f"Disable {key} (default: enabled)",
                    )
                else:
                    parser.add_argument(
                        f"--{path}",
                        action="store_true",
                        default=argparse.SUPPRESS,
                        help=f"Enable {key} (default: disabled)",
                    )
            else:
                options = (
                    {"nargs": "+"}
                    if isinstance(value, Container) and not isinstance(value, str)
                    else {}
                )
                parser.add_argument(
                    f"--{path}",
                    type=str,  # Pydantic casts and validates the CLI string.
                    default=argparse.SUPPRESS,
                    help=f"Set {key} (default: {value})",
                    **options,
                )

    register_fields(default_model, parent_key)
    return dynamic_kwargs_paths


def _parse_dynamic_cli_value(value: str) -> Any:
    """Accept JSON values for new class kwargs, leaving ordinary strings unchanged."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def add_new_class_kwarg_cli_args(
    parser: argparse.ArgumentParser,
    cli_args: list[str],
    dynamic_kwargs_paths: set[str],
) -> None:
    """Register previously absent keys beneath known dynamic class kwargs paths."""
    _, unknown_args = parser.parse_known_args(cli_args)
    added_options: set[str] = set()
    for arg in unknown_args:
        if not arg.startswith("--"):
            continue
        option = arg.split("=", 1)[0]
        path = option.removeprefix("--")
        if option in added_options:
            continue
        if any(
            path.startswith(f"{prefix}.") and path != f"{prefix}."
            for prefix in dynamic_kwargs_paths
        ):
            parser.add_argument(
                option, type=_parse_dynamic_cli_value, default=argparse.SUPPRESS
            )
            added_options.add(option)
