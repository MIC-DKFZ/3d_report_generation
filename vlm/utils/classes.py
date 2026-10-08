"""Nested attribute access and Python class discovery."""

import importlib
import os
import pkgutil
from typing import Any


def getattr_nested(obj, attr_path: str) -> Any:
    parts = attr_path.split(".")
    for part in parts:
        obj = getattr(obj, part)
    return obj


def setattr_nested(obj, attr_path: str, value) -> None:
    parts = attr_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def get_class_from_registry(class_name: str, registry: list[type]) -> type | None:
    candidates = [c for c in registry if c.__name__ == class_name]
    if candidates:
        return candidates[0]
    return None


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(os.path.join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr
