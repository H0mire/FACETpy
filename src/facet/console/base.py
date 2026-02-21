"""Base classes and enums for FACETpy console rendering."""

from __future__ import annotations

import contextlib
from abc import ABC
from enum import Enum
from typing import Any, Dict, Generator, Optional, List


class ConsoleMode(str, Enum):
    """Supported console rendering strategies."""

    MODERN = "modern"
    CLASSIC = "classic"


class BaseConsole(ABC):
    """Interface describing the capabilities of a console renderer."""

    enabled: bool = False
    requires_sink: bool = False

    def set_pipeline_metadata(self, metadata: Dict[str, Any]) -> None:
        """Store contextual metadata shown alongside progress details."""

    def start_pipeline(
        self,
        name: str,
        total_steps: int,
        step_names: Optional[List[str]] = None,
    ) -> None:
        """Begin tracking a pipeline run."""

    def step_started(self, index: int, name: str) -> None:
        """Mark a processor as running."""

    def step_completed(
        self,
        index: int,
        name: str,
        duration: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a processor as finished."""

    def pipeline_complete(self, success: bool, duration: float) -> None:
        """Finalize the pipeline view when execution completes."""

    def pipeline_failed(
        self,
        duration: float,
        error: Exception,
        step_index: Optional[int],
        step_name: Optional[str],
    ) -> None:
        """Show failure state for the pipeline."""

    def update_step_progress(
        self,
        index: int,
        completed: float,
        total: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update the progress metadata for an in-flight processor."""

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

    def shutdown(self) -> None:
        """Release console resources (if any)."""


class NullConsole(BaseConsole):
    """No-op console used when fancy rendering is disabled."""

    enabled = False
    requires_sink = False

    def set_pipeline_metadata(self, metadata: Dict[str, Any]) -> None:  # noqa: D401
        return None

    def start_pipeline(
        self,
        name: str,
        total_steps: int,
        step_names: Optional[List[str]] = None,
    ) -> None:  # noqa: D401
        return None

    def step_started(self, index: int, name: str) -> None:  # noqa: D401
        return None

    def step_completed(
        self,
        index: int,
        name: str,
        duration: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:  # noqa: D401
        return None

    def pipeline_complete(self, success: bool, duration: float) -> None:  # noqa: D401
        return None

    def pipeline_failed(
        self,
        duration: float,
        error: Exception,
        step_index: Optional[int],
        step_name: Optional[str],
    ) -> None:  # noqa: D401
        return None

    def update_step_progress(
        self,
        index: int,
        completed: float,
        total: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:  # noqa: D401
        return None

    def log_sink(self, message: Any) -> None:  # noqa: D401
        return None

    def get_rich_console(self) -> Any:  # noqa: D401
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
