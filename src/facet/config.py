"""Central configuration handling for FACETpy.

Configuration precedence (highest to lowest):
1. Runtime overrides via :func:`set_config`
2. Environment variables
3. Global config file (TOML)
4. Built-in defaults
"""

from __future__ import annotations

import contextlib
import os
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}
_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
_CONSOLE_ALIASES = {"legacy": "classic", "loguru": "classic"}

_CONFIG_DEFAULTS: dict[str, Any] = {
    "console_mode": "classic",
    "log_level": "INFO",
    "log_file": False,
    "log_file_level": "DEBUG",
    "log_dir": None,
    "auto_logging": True,
}

_RUNTIME_OVERRIDES: dict[str, Any] = {}


def _default_config_file_path() -> Path:
    return Path.home() / ".config" / "facetpy" / "config.toml"


def _parse_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUTHY:
            return True
        if lowered in _FALSEY:
            return False
    raise ValueError(f"Invalid boolean value for '{key}': {value!r}")


def _normalize_console_mode(value: Any) -> str:
    normalized = str(value).strip().lower()
    normalized = _CONSOLE_ALIASES.get(normalized, normalized)
    if normalized not in {"classic", "modern"}:
        raise ValueError("console_mode must be one of: classic, modern")
    return normalized


def _normalize_log_level(value: Any, key: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in _LOG_LEVELS:
        raise ValueError(f"{key} must be one of: {', '.join(sorted(_LOG_LEVELS))}")
    return normalized


def _normalize_config_values(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    if "console_mode" in values:
        normalized["console_mode"] = _normalize_console_mode(values["console_mode"])
    if "log_level" in values:
        normalized["log_level"] = _normalize_log_level(values["log_level"], "log_level")
    if "log_file" in values:
        normalized["log_file"] = _parse_bool(values["log_file"], "log_file")
    if "log_file_level" in values:
        normalized["log_file_level"] = _normalize_log_level(values["log_file_level"], "log_file_level")
    if "log_dir" in values:
        log_dir = values["log_dir"]
        normalized["log_dir"] = None if log_dir in (None, "") else str(log_dir)
    if "auto_logging" in values:
        normalized["auto_logging"] = _parse_bool(values["auto_logging"], "auto_logging")

    return normalized


def _normalize_nonfatal(values: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize config values key-by-key, ignoring invalid entries."""
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        try:
            normalized.update(_normalize_config_values({key: value}))
        except ValueError:
            continue
    return normalized


def _extract_file_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    section = raw.get("facet", raw)
    if not isinstance(section, Mapping):
        return {}

    result: dict[str, Any] = {}
    for key in _CONFIG_DEFAULTS:
        if key in section:
            result[key] = section[key]

    # Optional nested form:
    # [facet.logging]
    # level = "INFO"
    # file_enabled = true
    logging_section = section.get("logging")
    if isinstance(logging_section, Mapping):
        if "level" in logging_section:
            result["log_level"] = logging_section["level"]
        if "file_enabled" in logging_section:
            result["log_file"] = logging_section["file_enabled"]
        if "file_level" in logging_section:
            result["log_file_level"] = logging_section["file_level"]
        if "dir" in logging_section:
            result["log_dir"] = logging_section["dir"]
        if "console_mode" in logging_section:
            result["console_mode"] = logging_section["console_mode"]

    return result


def _load_file_config() -> dict[str, Any]:
    path_value = os.environ.get("FACET_CONFIG_FILE")
    config_path = Path(path_value).expanduser() if path_value else _default_config_file_path()
    if not config_path.exists():
        return {}

    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except Exception:
        return {}

    return _normalize_nonfatal(_extract_file_config(data))


def _load_env_config() -> dict[str, Any]:
    values: dict[str, Any] = {}
    env = os.environ

    if "FACET_CONSOLE_MODE" in env:
        values["console_mode"] = env["FACET_CONSOLE_MODE"]
    if "FACET_LOG_CONSOLE_LEVEL" in env:
        values["log_level"] = env["FACET_LOG_CONSOLE_LEVEL"]
    if "FACET_LOG_FILE" in env:
        values["log_file"] = env["FACET_LOG_FILE"]
    if "FACET_LOG_FILE_LEVEL" in env:
        values["log_file_level"] = env["FACET_LOG_FILE_LEVEL"]
    if "FACET_LOG_DIR" in env:
        values["log_dir"] = env["FACET_LOG_DIR"]
    if "FACET_DISABLE_AUTO_LOGGING" in env:
        disable_val = env["FACET_DISABLE_AUTO_LOGGING"].strip().lower()
        if disable_val in _TRUTHY:
            values["auto_logging"] = False
        elif disable_val in _FALSEY:
            values["auto_logging"] = True

    return _normalize_nonfatal(values)


def _resolve_config() -> dict[str, Any]:
    resolved = dict(_CONFIG_DEFAULTS)
    resolved.update(_load_file_config())
    resolved.update(_load_env_config())
    resolved.update(_RUNTIME_OVERRIDES)
    return resolved


def get_config(key: str | None = None) -> Any:
    """Return the resolved FACETpy configuration.

    Parameters
    ----------
    key : str, optional
        Optional key to retrieve. If ``None``, the full config dict is returned.
    """
    resolved = _resolve_config()
    if key is None:
        return deepcopy(resolved)
    if key not in resolved:
        raise KeyError(f"Unknown FACETpy config key: {key}")
    return deepcopy(resolved[key])


def _reconfigure_logging_if_available() -> None:
    with contextlib.suppress(Exception):
        from .logging_config import configure_logging

        configure_logging(force=True)


def set_config(config: Mapping[str, Any] | None = None, /, *, apply_logging: bool = True, **kwargs) -> dict[str, Any]:
    """Set in-process runtime config overrides (highest precedence).

    Parameters
    ----------
    config : mapping, optional
        Optional mapping with config keys/values.
    apply_logging : bool
        Reconfigure FACETpy logging immediately after applying overrides.
    **kwargs
        Additional key/value pairs (merged with ``config``).
    """
    values: dict[str, Any] = {}
    if config is not None:
        values.update(dict(config))
    values.update(kwargs)

    unknown_keys = sorted(set(values) - set(_CONFIG_DEFAULTS))
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise KeyError(f"Unknown FACETpy config key(s): {joined}")

    normalized = _normalize_config_values(values)
    _RUNTIME_OVERRIDES.update(normalized)

    if apply_logging:
        _reconfigure_logging_if_available()

    return get_config()


def reset_config(*, apply_logging: bool = True) -> dict[str, Any]:
    """Clear runtime overrides and return the resolved config."""
    _RUNTIME_OVERRIDES.clear()
    if apply_logging:
        _reconfigure_logging_if_available()
    return get_config()


__all__ = ["get_config", "set_config", "reset_config"]
