"""Lightweight helpers for processor-level progress updates."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from .manager import get_console

_CURRENT_STEP_INDEX: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "facet_console_current_step_index", default=None
)


def set_current_step_index(index: Optional[int]) -> None:
    """Store the index of the processor currently executing on this thread."""

    _CURRENT_STEP_INDEX.set(index)


def get_current_step_index() -> Optional[int]:
    """Return the index of the processor currently executing on this thread."""

    return _CURRENT_STEP_INDEX.get()


@dataclass
class ProcessorProgress:
    """Convenience wrapper for pushing progress updates to the console."""

    total: Optional[float] = None
    message: Optional[str] = None
    current: float = 0.0
    index: Optional[int] = field(default=None, init=False, repr=False)
    console: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.index = get_current_step_index()
        self.console = get_console()
        if self.index is not None:
            self.console.update_step_progress(
                self.index,
                completed=self.current,
                total=self.total,
                message=self.message,
            )

    def advance(self, amount: float = 1.0, message: Optional[str] = None) -> None:
        """Advance by ``amount`` units and optionally update the message."""

        self.current += amount
        self._sync(message)

    def update(
        self,
        current: Optional[float] = None,
        total: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        """Set the absolute progress values."""

        if current is not None:
            self.current = current
        if total is not None:
            self.total = total
        self._sync(message)

    def complete(self, message: Optional[str] = None) -> None:
        """Mark the progress as complete."""

        if self.total is not None:
            self.current = self.total
        self._sync(message)

    # Context manager helpers -------------------------------------------------
    def __enter__(self) -> "ProcessorProgress":  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        if exc_type is None:
            self.complete()

    # Internal helpers --------------------------------------------------------
    def _sync(self, message: Optional[str]) -> None:
        if self.index is None:
            return
        self.console.update_step_progress(
            self.index,
            completed=self.current,
            total=self.total,
            message=message,
        )


def processor_progress(total: Optional[float] = None, message: Optional[str] = None) -> ProcessorProgress:
    """Factory helper so processors can do ``with processor_progress(...)``."""

    return ProcessorProgress(total=total, message=message)


def report_metric(
    name: str,
    value: Any,
    *,
    label: Optional[str] = None,
    display: Optional[str] = None,
    level: str = "INFO",
) -> None:
    """Expose metrics inside the live console *and* log them via loguru.

    ``label`` overrides the key used in the UI (defaults to ``name``).
    ``display`` lets callers control formatting (e.g., ``"{value:.2f}"``).
    """

    console = get_console()
    text_value = display if display is not None else _format_metric_value(value)
    metric_label = label or name
    try:
        console.set_pipeline_metadata({metric_label: text_value})
    except Exception:
        # Rendering isn't critical for the actual logging signal.
        logger.opt(exception=True).debug("Failed to push metric {} to console", metric_label)
    readable_label = metric_label.replace("_", " ")
    logger.log(level.upper(), f"{readable_label}: {text_value}")


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (int, bool)):
        return str(value)
    return str(value)
