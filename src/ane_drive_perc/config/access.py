from collections.abc import Mapping
from pathlib import Path
from typing import Any


Config = dict[str, Any]
ConfigMapping = Mapping[str, Any]


def expect_mapping(mapping: ConfigMapping, key: str) -> Config:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Expected {key!r} to be a mapping.")
    return value


def expect_dict(mapping: ConfigMapping, key: str) -> Config:
    return expect_mapping(mapping, key)


def expect_str(mapping: ConfigMapping, key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Expected {key!r} to be a string.")
    return value


def expect_optional_str(mapping: ConfigMapping, key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Expected {key!r} to be a string or null.")
    return value


def optional_str(mapping: ConfigMapping, key: str) -> str | None:
    """Alias for expect_optional_str."""
    return expect_optional_str(mapping, key)


def expect_int(mapping: ConfigMapping, key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Expected {key!r} to be an integer.")
    return value


def expect_float(mapping: ConfigMapping, key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"Expected {key!r} to be numeric.")
    return float(value)


def expect_bool(mapping: ConfigMapping, key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"Expected {key!r} to be a boolean.")
    return value


def get_bool(mapping: ConfigMapping, key: str, *, default: bool) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"Expected {key!r} to be a boolean.")
    return value


def get_str(mapping: ConfigMapping, key: str, *, default: str) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"Expected {key!r} to be a string.")
    return value


def get_int(mapping: ConfigMapping, key: str, *, default: int) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Expected {key!r} to be an integer.")
    return value


def get_float(mapping: ConfigMapping, key: str, *, default: float) -> float:
    value = mapping.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"Expected {key!r} to be numeric.")
    return float(value)


def get_path(
    mapping: ConfigMapping,
    key: str,
    *,
    default: str | Path | None = None,
) -> Path:
    value = mapping.get(key, default)
    if value is None:
        raise TypeError(f"Expected {key!r} to be a path-like string.")
    if not isinstance(value, str | Path):
        raise TypeError(f"Expected {key!r} to be a path-like string.")
    return Path(value)
