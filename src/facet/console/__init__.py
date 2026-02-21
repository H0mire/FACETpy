"""Console rendering helpers for FACETpy."""

import contextlib
from typing import Generator

from .base import BaseConsole, ConsoleMode, NullConsole
from .manager import configure_console, get_console
from .progress import ProcessorProgress, processor_progress, report_metric


@contextlib.contextmanager
def suspend_raw_mode() -> Generator[None, None, None]:
    """Context manager that temporarily suspends raw terminal mode.

    Delegates to the currently active console's ``suspend_raw_mode()``
    implementation.  When the classic console (``NullConsole``) is active the
    context manager is a no-op, so callers don't need to check the mode.

    Usage::

        from facet.console import suspend_raw_mode

        with suspend_raw_mode():
            answer = input("Continue? [y/N] ")
    """
    with get_console().suspend_raw_mode():
        yield


__all__ = [
    "BaseConsole",
    "ConsoleMode",
    "NullConsole",
    "configure_console",
    "get_console",
    "suspend_raw_mode",
    "ProcessorProgress",
    "processor_progress",
    "report_metric",
]
