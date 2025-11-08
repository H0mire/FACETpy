"""Rich-powered console renderer for FACETpy."""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from .base import BaseConsole

try:  # Local import guard so the manager can fall back gracefully.
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - falls back to classic logging.
    Console = None  # type: ignore
    Live = None  # type: ignore
    Group = None  # type: ignore
    Layout = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore
    Text = None  # type: ignore
    Align = None  # type: ignore
    box = None  # type: ignore
    _RICH_AVAILABLE = False


@dataclass
class StepState:
    """Keeps track of processor progress for the live console."""

    index: int
    name: str = "–"
    status: str = "pending"
    duration: float = 0.0
    started_at: Optional[float] = None
    progress_value: float = 0.0
    progress_total: Optional[float] = None
    progress_message: str = ""


class ModernConsole(BaseConsole):
    """Interactive console that renders FACETpy progress using Rich."""

    enabled = True
    requires_sink = True

    def __init__(self) -> None:
        if not _RICH_AVAILABLE:
            raise RuntimeError("Rich is not available")

        self.console = Console(highlight=False)
        if not self.console.is_terminal or os.environ.get("TERM") == "dumb":
            raise RuntimeError("Interactive console requires a TTY terminal")

        self.pipeline_name: str = "FACETpy Pipeline"
        self.pipeline_status: str = "Idle"
        self.pipeline_start: Optional[float] = None
        self.total_steps: int = 0
        self.completed_steps: int = 0
        self.current_step_name: str = "–"
        self.last_step_name: str = "–"
        self.last_step_duration: Optional[float] = None
        self.step_states: List[StepState] = []
        self.extra_metrics: Dict[str, str] = {}
        self.log_messages: Deque[Dict[str, str]] = deque()

        self._lock = threading.Lock()
        self._ticker_stop = threading.Event()
        self._ticker_thread: Optional[threading.Thread] = None
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            auto_refresh=False,
        )
        self._live.start()
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Lifecyle helpers
    # ------------------------------------------------------------------
    @staticmethod
    def is_supported() -> bool:
        if not _RICH_AVAILABLE:
            return False
        try:
            console = Console(highlight=False)
        except Exception:  # pragma: no cover - defensive guard
            return False
        if not console.is_terminal:
            return False
        return os.environ.get("TERM", "").lower() not in {"", "dumb"}

    def shutdown(self) -> None:
        with self._lock:
            self._stop_ticker_locked()
            if self._live is not None:
                try:
                    self._live.stop()
                except Exception:
                    pass
                self._live = None

    # ------------------------------------------------------------------
    # Public API used by the pipeline/logging configuration
    # ------------------------------------------------------------------
    def set_pipeline_metadata(self, metadata: Dict[str, Any]) -> None:
        with self._lock:
            for key, value in metadata.items():
                self.extra_metrics[key] = str(value)
            self._refresh_locked()

    def start_pipeline(
        self,
        name: str,
        total_steps: int,
        step_names: Optional[List[str]] = None,
    ) -> None:
        with self._lock:
            self.pipeline_name = name
            self.total_steps = max(total_steps, 0)
            self.completed_steps = 0
            self.pipeline_status = "Running"
            self.pipeline_start = time.time()
            self.current_step_name = "–"
            self.last_step_name = "–"
            self.last_step_duration = None
            self.extra_metrics = {}
            self.log_messages.clear()
            self.step_states = [
                StepState(
                    index=i,
                    name=(step_names[i] if step_names and i < len(step_names) else f"Step {i + 1}"),
                )
                for i in range(total_steps)
            ]
            self._start_ticker_locked()
            self._refresh_locked()

    def step_started(self, index: int, name: str) -> None:
        with self._lock:
            if not self._ensure_index(index):
                return
            state = self.step_states[index]
            state.name = name
            state.status = "running"
            state.started_at = time.time()
            state.progress_value = 0.0
            state.progress_total = None
            state.progress_message = ""
            self.current_step_name = name
            self._append_log("INFO", f"▶ {name}")
            self._refresh_locked()

    def step_completed(
        self,
        index: int,
        name: str,
        duration: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if not self._ensure_index(index):
                return
            state = self.step_states[index]
            state.name = name
            state.status = "done"
            state.duration = duration
            state.started_at = None
            if state.progress_total is not None:
                state.progress_value = state.progress_total
            state.progress_message = ""
            self.completed_steps = max(self.completed_steps, index + 1)
            self.last_step_name = name
            self.last_step_duration = duration
            if metrics:
                for key, value in metrics.items():
                    self.extra_metrics[key] = str(value)
            self._append_log("SUCCESS", f"✔ {name} ({duration:.2f}s)")
            self._refresh_locked()

    def pipeline_complete(self, success: bool, duration: float) -> None:
        with self._lock:
            self.pipeline_status = "Completed" if success else "Finished"
            self.extra_metrics["elapsed"] = f"{duration:.2f}s"
            self.current_step_name = "–"
            self._append_log("INFO", f"Pipeline completed in {duration:.2f}s")
            self._stop_ticker_locked()
            self._refresh_locked()

    def pipeline_failed(
        self,
        duration: float,
        error: Exception,
        step_index: Optional[int],
        step_name: Optional[str],
    ) -> None:
        with self._lock:
            self.pipeline_status = "Failed"
            self.extra_metrics["elapsed"] = f"{duration:.2f}s"
            if step_index is not None and self._ensure_index(step_index):
                state = self.step_states[step_index]
                state.status = "failed"
                if step_name:
                    state.name = step_name
            self._append_log("ERROR", f"✖ {type(error).__name__}: {error}")
            self._stop_ticker_locked()
            self._refresh_locked()

    def update_step_progress(
        self,
        index: int,
        completed: float,
        total: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        with self._lock:
            if not self._ensure_index(index):
                return
            state = self.step_states[index]
            state.progress_value = max(completed, 0.0)
            if total is not None:
                state.progress_total = max(total, 0.0)
            if message is not None:
                state.progress_message = message
            self._refresh_locked()

    def log_sink(self, message: Any) -> None:  # noqa: ANN401 (loguru message object)
        record = getattr(message, "record", {})
        level_obj = record.get("level")
        if hasattr(level_obj, "name"):
            level = level_obj.name
        elif isinstance(level_obj, dict):
            level = level_obj.get("name", "INFO")
        else:
            level = "INFO"
        text = record.get("message", str(message))
        time_str = record.get("time")
        timestamp = time_str.strftime("%H:%M:%S") if time_str else time.strftime("%H:%M:%S")
        with self._lock:
            self._append_log(level, text, timestamp)
            self._refresh_locked()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_log(self, level: str, text: str, timestamp: Optional[str] = None) -> None:
        if timestamp is None:
            timestamp = time.strftime("%H:%M:%S")
        self.log_messages.append(
            {
                "time": timestamp,
                "level": level.upper(),
                "style": self._level_color(level),
                "text": text.strip(),
            }
        )
        self._trim_logs()

    def _ensure_index(self, index: int) -> bool:
        if index < 0:
            return False
        if index >= len(self.step_states):
            # Extend list so long-running custom pipelines are still shown
            for i in range(len(self.step_states), index + 1):
                self.step_states.append(StepState(index=i, name=f"Step {i + 1}"))
        return True

    def _refresh_locked(self) -> None:
        if self._live is None:
            return
        renderable = self._render()
        self._live.update(renderable, refresh=True)

    def _start_ticker_locked(self) -> None:
        if self._ticker_thread and self._ticker_thread.is_alive():
            return
        self._ticker_stop.clear()
        self._ticker_thread = threading.Thread(target=self._ticker_loop, daemon=True)
        self._ticker_thread.start()

    def _stop_ticker_locked(self) -> None:
        if self._ticker_thread and self._ticker_thread.is_alive():
            self._ticker_stop.set()
            self._ticker_thread.join(timeout=0.2)
        self._ticker_thread = None

    def _ticker_loop(self) -> None:
        while not self._ticker_stop.wait(0.25):
            with self._lock:
                self._refresh_locked()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render(self):  # type: ignore[override]
        if Layout is None:  # pragma: no cover - guard for safety
            return ""
        layout = Layout(name="root")
        header_size, body_size, log_size = self._layout_sizes()
        layout.split_column(
            Layout(self._render_header(), name="header", size=header_size),
            Layout(name="body", size=body_size if body_size > 0 else None),
            Layout(self._render_logs_panel(), name="logs", size=log_size if log_size > 0 else None),
        )
        body = Layout(name="body")
        body.split_row(
            Layout(self._render_steps_panel(), name="steps", ratio=3),
            Layout(self._render_metrics_panel(), name="metrics", ratio=2),
        )
        layout["body"].update(body)
        return layout

    def _render_header(self) -> Panel:
        status_color = {
            "Idle": "grey62",
            "Running": "cyan",
            "Completed": "green",
            "Finished": "green",
            "Failed": "red",
        }.get(self.pipeline_status, "cyan")
        text = Text()
        text.append(self.pipeline_name or "FACETpy Pipeline", style="bold white")
        text.append("  ")
        text.append(self.pipeline_status, style=f"bold {status_color}")
        elapsed = self._elapsed_seconds()
        if elapsed > 0:
            text.append(f"  •  {elapsed:.1f}s elapsed", style="dim")
        return Panel(text, border_style="bright_black", padding=(0, 1))

    def _render_steps_panel(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column("Processor", style="bold", ratio=2)
        table.add_column("Status", style="white", width=12)
        table.add_column("Progress", style="white", width=24)
        table.add_column("Duration", style="white", width=10)

        if not self.step_states:
            table.add_row("Waiting for pipeline", "idle", "", "–")
        else:
            for step in self.step_states:
                icon, style = self._status_badge(step.status)
                duration = f"{step.duration:.2f}s" if step.duration else ("…" if step.status == "running" else "–")
                display_name = step.name or f"Step {step.index + 1}"
                table.add_row(
                    display_name,
                    f"[{style}]{icon} {step.status.capitalize()}[/]",
                    self._step_progress(step),
                    duration,
                )

        return Panel(table, title="Processors", border_style="bright_black", box=box.ROUNDED)

    def _render_metrics_panel(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column("Metric", style="bold", width=16)
        table.add_column("Value", style="white")

        table.add_row("Progress", self._progress_bar())
        table.add_row("Completed", f"{self.completed_steps}/{max(self.total_steps, 1)}")
        table.add_row("Current", self.current_step_name)
        last = f"{self.last_step_name} ({self.last_step_duration:.2f}s)" if self.last_step_duration else self.last_step_name
        table.add_row("Last", last)
        table.add_row("Elapsed", f"{self._elapsed_seconds():.1f}s")

        diagnostics, reported = self._split_extra_metrics()
        for key, value in diagnostics:
            table.add_row(key.replace("_", " ").title(), value)

        if reported:
            table.add_row("", "")
            table.add_row("[bold]Reported Metrics[/]", "")
            for key, value in reported:
                table.add_row(key.replace("_", " ").title(), value)

        return Panel(table, title="Metrics", border_style="bright_black", box=box.ROUNDED)

    def _render_logs_panel(self) -> Panel:
        self._trim_logs()
        if not self.log_messages:
            body: Any = Align.left(Text("Logs will appear as processors run", style="dim"))
        else:
            rows = []
            for entry in self.log_messages:
                row = Text()
                row.append(entry["time"], style="dim")
                row.append("  ")
                row.append(f"{entry['level']:>7}", style=entry["style"])
                row.append("  ")
                row.append(entry["text"])
                rows.append(row)
            body = Align.left(Group(*rows))
        return Panel(body, title="Live Logs", border_style="bright_black", box=box.ROUNDED)

    def _step_progress(self, step: StepState) -> str:
        if step.progress_total in (None, 0):
            return step.progress_message or ("…" if step.status == "running" else "")

        ratio = 0.0 if step.progress_total == 0 else min(max(step.progress_value / step.progress_total, 0.0), 1.0)
        bar = self._inline_bar(ratio, width=14)
        percent = int(ratio * 100)
        message = f" {percent:>3d}%"
        if step.progress_message:
            message += f" {step.progress_message}"
        return bar + message

    def _progress_bar(self) -> str:
        total = max(self.total_steps, 1)
        ratio = min(max(self.completed_steps / total, 0.0), 1.0)
        filled = int(ratio * 24)
        empty = 24 - filled
        bar = f"[green]{'█' * filled}[/][grey42]{'░' * empty}[/]"
        percent = int(ratio * 100)
        return f"{bar} {percent:>3d}%"

    @staticmethod
    def _inline_bar(ratio: float, width: int = 10) -> str:
        ratio = min(max(ratio, 0.0), 1.0)
        filled = int(ratio * width)
        empty = width - filled
        return f"[green]{'█' * filled}[/][grey42]{'░' * empty}[/]"

    def _elapsed_seconds(self) -> float:
        if not self.pipeline_start:
            return 0.0
        return max(time.time() - self.pipeline_start, 0.0)

    @staticmethod
    def _level_color(level: str) -> str:
        mapping = {
            "DEBUG": "cyan",
            "INFO": "bright_white",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bright_red",
        }
        return mapping.get(level.upper(), "white")

    @staticmethod
    def _status_badge(status: str) -> tuple[str, str]:
        status_normalized = status.lower()
        if status_normalized == "running":
            return "▶", "cyan"
        if status_normalized == "done" or status_normalized == "completed":
            return "✔", "green"
        if status_normalized == "failed":
            return "✖", "red"
        return "●", "grey62"

    def _layout_sizes(self) -> tuple[int, int, int]:
        header_size = 3
        min_log_size = 6
        total_height = self._terminal_height()
        available = max(total_height - header_size, 0)
        body_needed = self._body_panel_height_needed()

        if available <= 0:
            return header_size, 0, 0

        if available >= body_needed + min_log_size:
            body_size = body_needed
            log_size = available - body_size
        else:
            body_size = min(body_needed, available)
            log_size = available - body_size

        return header_size, body_size, max(log_size, 0)

    def _body_panel_height_needed(self) -> int:
        processors_height = self._processors_panel_height_needed()
        metrics_height = self._metrics_panel_height_needed()
        return max(processors_height, metrics_height)

    def _processors_panel_height_needed(self) -> int:
        rows = len(self.step_states) if self.step_states else 1
        panel_chrome = 4  # borders + header rows
        return rows + panel_chrome

    def _metrics_panel_height_needed(self) -> int:
        base_rows = 5  # progress + completed + current + last + elapsed
        diagnostics, reported = self._split_extra_metrics()
        header_rows = 2 if reported else 0  # spacer + header label
        extra_rows = len(diagnostics) + len(reported)
        panel_chrome = 4
        return base_rows + header_rows + extra_rows + panel_chrome

    def _split_extra_metrics(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        diagnostics: list[tuple[str, str]] = []
        reported: list[tuple[str, str]] = []
        for key, value in self.extra_metrics.items():
            if key == "elapsed":
                continue
            if key in {"execution_mode", "n_jobs", "last_duration"}:
                diagnostics.append((key, value))
            else:
                reported.append((key, value))
        return diagnostics, reported

    def _terminal_height(self) -> int:
        if self.console is None:
            return 24
        try:
            return int(self.console.size.height)
        except Exception:
            return 24

    def _trim_logs(self) -> None:
        capacity = self._log_capacity()
        if capacity <= 0:
            self.log_messages.clear()
            return
        while len(self.log_messages) > capacity:
            self.log_messages.popleft()

    def _log_capacity(self) -> int:
        _, _, log_size = self._layout_sizes()
        if log_size <= 0:
            return 0
        panel_chrome = 4
        return max(1, log_size - panel_chrome)
