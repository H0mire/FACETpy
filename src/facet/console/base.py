"""Base classes and enums for FACETpy console rendering."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Generator
from enum import StrEnum
from typing import Any


class ConsoleMode(StrEnum):
    """Supported console rendering strategies."""

    MODERN = "modern"
    CLASSIC = "classic"


class BaseConsole(ABC):
    """Interface describing the capabilities of a console renderer."""

    enabled: bool = False
    requires_sink: bool = False

    @abstractmethod
    def set_pipeline_metadata(self, metadata: dict[str, Any]) -> None:
        """Store contextual metadata shown alongside progress details."""

    @abstractmethod
    def start_pipeline(
        self,
        name: str,
        total_steps: int,
        step_names: list[str] | None = None,
    ) -> None:
        """Begin tracking a pipeline run."""

    @abstractmethod
    def step_started(self, index: int, name: str) -> None:
        """Mark a processor as running."""

    @abstractmethod
    def step_completed(
        self,
        index: int,
        name: str,
        duration: float,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Mark a processor as finished."""

    @abstractmethod
    def pipeline_complete(self, success: bool, duration: float) -> None:
        """Finalize the pipeline view when execution completes."""

    @abstractmethod
    def pipeline_failed(
        self,
        duration: float,
        error: Exception,
        step_index: int | None,
        step_name: str | None,
    ) -> None:
        """Show failure state for the pipeline."""

    @abstractmethod
    def update_step_progress(
        self,
        index: int,
        completed: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        """Update the progress metadata for an in-flight processor."""

    # Channel-sequential batch lifecycle
    @abstractmethod
    def start_channel_batch(
        self,
        processor_names: list[str],
        channel_names: list[str],
        batch_step_offset: int = 0,
    ) -> None:
        """Enter channel-sequential display mode."""

    @abstractmethod
    def channel_started(self, ch_index: int, ch_name: str) -> None:
        """Mark a data channel as actively processing."""

    @abstractmethod
    def channel_processor_started(self, ch_index: int, proc_index: int) -> None:
        """Mark which processor is running within the current channel."""

    @abstractmethod
    def channel_processor_completed(
        self,
        ch_index: int,
        proc_index: int,
        duration: float,
        skipped: bool = False,
    ) -> None:
        """Mark a processor within a channel as complete (or skipped)."""

    @abstractmethod
    def channel_completed(self, ch_index: int, duration: float) -> None:
        """Mark a channel as fully processed."""

    @abstractmethod
    def end_channel_batch(self) -> None:
        """Exit channel-sequential display mode."""

    @abstractmethod
    def log_sink(self, message: Any) -> None:
        """Optional log sink wired into loguru."""

    def get_rich_console(self) -> Any:
        """Return a Rich Console that bypasses any output interception.

        When the modern live display is active, printing through
        ``sys.stdout`` would be captured by the log tee and garbled.
        This method returns a Console wired to the real TTY so callers
        (e.g. ``PipelineResult.print_metrics``) can render Rich panels
        cleanly.  Returns ``None`` when no special routing is needed.
        """
        return None

    @contextlib.contextmanager
    def suspend_raw_mode(self) -> Generator[None, None, None]:
        """Temporarily restore cooked terminal mode for input()/readline() calls.

        The default implementation is a no-op; ``ModernConsole`` overrides it
        to pause the raw-mode keyboard listener for the duration of the block.
        """
        yield

    @abstractmethod
    def shutdown(self) -> None:
        """Release console resources (if any)."""


class NullConsole(BaseConsole):
    """No-op console used when fancy rendering is disabled."""

    enabled = False
    requires_sink = False

    def set_pipeline_metadata(self, metadata: dict[str, Any]) -> None:  # noqa: D401
        return None

    def start_pipeline(
        self,
        name: str,
        total_steps: int,
        step_names: list[str] | None = None,
    ) -> None:  # noqa: D401
        return None

    def step_started(self, index: int, name: str) -> None:  # noqa: D401
        return None

    def step_completed(
        self,
        index: int,
        name: str,
        duration: float,
        metrics: dict[str, Any] | None = None,
    ) -> None:  # noqa: D401
        return None

    def pipeline_complete(self, success: bool, duration: float) -> None:  # noqa: D401
        return None

    def pipeline_failed(
        self,
        duration: float,
        error: Exception,
        step_index: int | None,
        step_name: str | None,
    ) -> None:  # noqa: D401
        return None

    def update_step_progress(
        self,
        index: int,
        completed: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:  # noqa: D401
        return None

    def log_sink(self, message: Any) -> None:  # noqa: D401
        return None

    def get_rich_console(self) -> Any:  # noqa: D401
        return None

    def start_channel_batch(
        self, processor_names: list[str], channel_names: list[str], batch_step_offset: int = 0
    ) -> None:  # noqa: D401
        return None

    def channel_started(self, ch_index: int, ch_name: str) -> None:  # noqa: D401
        return None

    def channel_processor_started(self, ch_index: int, proc_index: int) -> None:  # noqa: D401
        return None

    def channel_processor_completed(
        self, ch_index: int, proc_index: int, duration: float, skipped: bool = False
    ) -> None:  # noqa: D401
        return None

    def channel_completed(self, ch_index: int, duration: float) -> None:  # noqa: D401
        return None

    def end_channel_batch(self) -> None:  # noqa: D401
        return None

    def set_active_prompt(self, message: str) -> None:  # noqa: D401
        return None

    def clear_active_prompt(self) -> None:  # noqa: D401
        return None

    @contextlib.contextmanager
    def suspend_raw_mode(self) -> Generator[None, None, None]:  # noqa: D401
        yield

    def shutdown(self) -> None:  # noqa: D401
        return None
