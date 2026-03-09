"""Console manager that wires logging to the appropriate renderer."""

from __future__ import annotations

import contextlib

from .base import BaseConsole, ConsoleMode, NullConsole

_console: BaseConsole = NullConsole()


def configure_console(mode: str | ConsoleMode) -> BaseConsole:
    """Configure and return the global console instance."""
    global _console

    requested = ConsoleMode(mode) if not isinstance(mode, ConsoleMode) else mode

    if requested is ConsoleMode.MODERN:
        console = _build_modern_console()
        if console is not None:
            _replace_console(console)
            return _console

    _replace_console(NullConsole())
    return _console


def get_console() -> BaseConsole:
    """Return the currently configured console (may be a NullConsole)."""
    return _console


def _build_modern_console() -> BaseConsole | None:
    try:
        from .modern import ModernConsole
    except Exception:
        return None

    if not ModernConsole.is_supported():
        return None

    return ModernConsole()


def _replace_console(new_console: BaseConsole) -> None:
    global _console

    if _console is new_console:
        return

    with contextlib.suppress(Exception):
        _console.shutdown()

    _console = new_console
