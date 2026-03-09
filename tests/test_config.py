"""Tests for FACETpy central configuration handling."""

from __future__ import annotations

import pytest

from facet.config import get_config, reset_config, set_config

_FACET_ENV_KEYS = [
    "FACET_CONFIG_FILE",
    "FACET_CONSOLE_MODE",
    "FACET_LOG_CONSOLE_LEVEL",
    "FACET_LOG_FILE",
    "FACET_LOG_FILE_LEVEL",
    "FACET_LOG_DIR",
    "FACET_DISABLE_AUTO_LOGGING",
]


@pytest.fixture(autouse=True)
def _clean_config(monkeypatch):
    for key in _FACET_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    reset_config(apply_logging=False)
    yield
    reset_config(apply_logging=False)


@pytest.mark.unit
def test_defaults_are_available():
    cfg = get_config()

    assert cfg["console_mode"] == "classic"
    assert cfg["log_level"] == "INFO"
    assert cfg["log_file"] is False
    assert cfg["auto_logging"] is True


@pytest.mark.unit
def test_file_config_is_loaded(temp_dir, monkeypatch):
    config_file = temp_dir / "facet.toml"
    config_file.write_text(
        """
[facet]
console_mode = "modern"
log_level = "DEBUG"
log_file = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FACET_CONFIG_FILE", str(config_file))

    cfg = get_config()
    assert cfg["console_mode"] == "modern"
    assert cfg["log_level"] == "DEBUG"
    assert cfg["log_file"] is True


@pytest.mark.unit
def test_env_overrides_file_config(temp_dir, monkeypatch):
    config_file = temp_dir / "facet.toml"
    config_file.write_text(
        """
[facet]
log_level = "DEBUG"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FACET_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("FACET_LOG_CONSOLE_LEVEL", "WARNING")

    assert get_config("log_level") == "WARNING"


@pytest.mark.unit
def test_runtime_overrides_env(monkeypatch):
    monkeypatch.setenv("FACET_LOG_CONSOLE_LEVEL", "WARNING")

    set_config(log_level="ERROR", apply_logging=False)
    assert get_config("log_level") == "ERROR"


@pytest.mark.unit
def test_reset_config_clears_runtime_override(monkeypatch):
    monkeypatch.setenv("FACET_LOG_CONSOLE_LEVEL", "WARNING")

    set_config(log_level="ERROR", apply_logging=False)
    assert get_config("log_level") == "ERROR"

    reset_config(apply_logging=False)
    assert get_config("log_level") == "WARNING"


@pytest.mark.unit
def test_console_mode_alias_legacy_maps_to_classic():
    set_config(console_mode="legacy", apply_logging=False)
    assert get_config("console_mode") == "classic"


@pytest.mark.unit
def test_invalid_env_values_are_ignored(monkeypatch):
    monkeypatch.setenv("FACET_LOG_FILE", "not-a-bool")
    monkeypatch.setenv("FACET_LOG_CONSOLE_LEVEL", "not-a-level")

    cfg = get_config()
    assert cfg["log_file"] is False
    assert cfg["log_level"] == "INFO"


@pytest.mark.unit
def test_set_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        set_config(nonexistent_key="value", apply_logging=False)
