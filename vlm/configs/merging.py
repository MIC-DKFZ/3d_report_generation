from typing import Any


def unflatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Turn dotted keys into nested dictionaries."""
    nested: dict[str, Any] = {}
    grouped: dict[str, dict[str, Any]] = {}
    for key, value in d.items():
        head, separator, tail = key.partition(".")
        if separator:
            grouped.setdefault(head, {})[tail] = value
        else:
            nested[head] = value

    for head, values in grouped.items():
        nested[head] = unflatten_dict(values)
    return nested


def recursive_dict_merge(d1: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values from d2 override those from d1.
    Nested dictionaries are merged recursively, all other values are replaced.
    """
    merged = dict(d1)
    for key, value in d2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = recursive_dict_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_overrides(
    config: dict[str, Any], overrides: dict[str, Any], effective: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge 'overrides' into the unvalidated 'config', like 'recursive_dict_merge'.

    A nested override of a field that 'config' leaves to its default starts from that field's
    'effective' value, the validated config's dump, so its other keys keep their defaults.
    """
    merged = dict(config)
    for key, value in overrides.items():
        base = config[key] if key in config else effective.get(key)
        if isinstance(base, dict) and isinstance(value, dict):
            value = merge_overrides(base, value, effective.get(key) or {})
        merged[key] = value
    return merged
