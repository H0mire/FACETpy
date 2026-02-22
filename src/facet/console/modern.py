"""Rich-powered console renderer for FACETpy."""

from __future__ import annotations

import atexit
import contextlib
import copy
import io
import os
import re
import signal
import sys
import threading
import time
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, TextIO

from .base import BaseConsole

# How long (seconds) to keep the display open after the pipeline finishes
# before closing automatically.  Override with FACET_CONSOLE_CLOSE_TIMEOUT.
_WAIT_FOR_CLOSE_TIMEOUT = float(os.environ.get("FACET_CONSOLE_CLOSE_TIMEOUT", "300"))

# Strip ANSI escape sequences (e.g. \x1b[31m) from terminal output
_ANSI_STRIP = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

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

# Check termios availability (POSIX only)
try:
    import select as _select
    import termios as _termios

    _TERMIOS_AVAILABLE = True
except ImportError:
    _TERMIOS_AVAILABLE = False


@dataclass
class StepState:
    """Keeps track of processor progress for the live console."""

    index: int
    name: str = "–"
    status: str = "pending"
    duration: float = 0.0
    started_at: float | None = None
    progress_value: float = 0.0
    progress_total: float | None = None
    progress_message: str = ""


@dataclass
class ChannelBatchState:
    """Live state for a channel-sequential execution batch."""

    active: bool = False
    processor_names: list[str] = field(default_factory=list)
    channel_names: list[str] = field(default_factory=list)
    batch_step_offset: int = 0

    current_ch_index: int = -1
    current_ch_name: str = ""
    current_proc_index: int = -1
    current_proc_start: float | None = None

    total_channels: int = 0
    completed_channels: int = 0

    proc_statuses: list[str] = field(default_factory=list)
    proc_durations: list[float | None] = field(default_factory=list)

    batch_start_time: float | None = None
    channel_start_time: float | None = None
    channel_durations: list[float] = field(default_factory=list)


class _LogTee(io.TextIOBase):
    """Redirects stdout/stderr writes to the Live Logs panel."""

    def __init__(
        self,
        real: TextIO,
        console: ModernConsole,
        level: str = "INFO",
    ) -> None:
        self._real = real
        self._console = console
        self._level = level
        self._buffer: list[str] = []

    # Rich Console queries these attributes to decide how to render.
    @property
    def encoding(self) -> str:
        return getattr(self._real, "encoding", "utf-8") or "utf-8"

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError(f"write() argument must be str, not {type(text).__name__}")
        lines: list[str] = []
        remaining = text
        while remaining:
            line, new_line, remaining = remaining.partition("\n")
            if new_line:
                self._buffer.append(line)
                raw = "".join(self._buffer)
                self._buffer.clear()
                # Only skip truly empty lines (strip ANSI for the blank-check only)
                if _ANSI_STRIP.sub("", raw).strip():
                    lines.append(raw)
            else:
                self._buffer.append(line)
                break
        for raw in lines:
            self._console._tee_append_log(self._level, raw)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            raw = "".join(self._buffer)
            self._buffer.clear()
            if _ANSI_STRIP.sub("", raw).strip():
                self._console._tee_append_log(self._level, raw)


class ModernConsole(BaseConsole):
    """Interactive console that renders FACETpy progress using Rich."""

    enabled = True
    requires_sink = True

    def __init__(self) -> None:
        if not _RICH_AVAILABLE:
            raise RuntimeError("Rich is not available")

        # Wire Console to real stdout *before* installing tee, so Live display
        # output bypasses the tee and avoids deadlock (tee acquires _lock).
        self._real_stdout = sys.stdout
        self._real_stderr = sys.stderr
        self.console = Console(file=self._real_stdout, highlight=False)
        if not self.console.is_terminal or os.environ.get("TERM") == "dumb":
            raise RuntimeError("Interactive console requires a TTY terminal")

        self.pipeline_name: str = "FACETpy Pipeline"
        self.pipeline_status: str = "Idle"
        self.pipeline_start: float | None = None
        self.total_steps: int = 0
        self.completed_steps: int = 0
        self.current_step_name: str = "–"
        self.last_step_name: str = "–"
        self.last_step_duration: float | None = None
        self.step_states: list[StepState] = []
        self.extra_metrics: dict[str, str] = {}

        # Rolling window shown in modern view (trimmed to fit terminal)
        self.log_messages: deque[dict[str, str]] = deque()
        # Complete unbounded history shared by the classic view
        self._full_log_history: list[dict[str, str]] = []

        # Channel-sequential batch state (active only while a batch runs)
        self._channel_batch = ChannelBatchState()

        # View mode: "modern" (dashboard) or "classic" (full-screen log)
        # Starts in classic so the user sees a clean log before the pipeline runs;
        # start_pipeline() switches to modern automatically.
        self._view_mode: str = "classic"
        # Lines scrolled up from the bottom in classic view (0 = pinned to bottom)
        self._classic_scroll_offset: int = 0

        # Set to True after pipeline_complete/pipeline_failed so the display
        # stays open until the user explicitly presses q.
        self._pipeline_done: bool = False
        self._quit_event = threading.Event()

        # Non-None while WaitForConfirmation (or similar) is blocking on input.
        # The footer replaces its normal hints with this text.
        self._active_prompt: str | None = None

        self._lock = threading.Lock()
        self._ticker_stop = threading.Event()
        self._ticker_thread: threading.Thread | None = None
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start()
        self._stdout_tee: _LogTee | None = None
        self._stderr_tee: _LogTee | None = None
        self._sink_console: Any | None = None  # set after tee is installed
        self._install_output_tee()

        # Keyboard listener (POSIX only)
        self._input_stop = threading.Event()
        self._input_thread: threading.Thread | None = None
        self._orig_term_settings: Any = None
        self._raw_mode_settings: Any = None  # saved after applying raw mode
        # Used by suspend_raw_mode() to pause the keyboard loop
        self._raw_suspend = threading.Event()
        self._raw_paused = threading.Event()
        self._mouse_tracking: bool = False
        self._start_keyboard_listener()

        # Signal handlers to restore terminal on SIGTERM / SIGHUP
        self._old_signal_handlers: dict[int, Any] = {}
        self._install_signal_handlers()

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
        # Always keep the display open until the user explicitly presses q.
        # Mark the pipeline as done (if not already) so the footer shows
        # "[q] close" and the keyboard loop routes q → _quit_event instead of SIGINT.
        if not self._quit_event.is_set():
            if not self._pipeline_done:
                with self._lock:
                    self._pipeline_done = True
                    self._refresh_locked()
            self._quit_event.wait(timeout=_WAIT_FOR_CLOSE_TIMEOUT)
        self._uninstall_signal_handlers()
        self._stop_keyboard_listener()
        self._uninstall_output_tee()
        with self._lock:
            self._stop_ticker_locked()
            if self._live is not None:
                with contextlib.suppress(Exception):
                    self._live.stop()
                self._live = None

    # ------------------------------------------------------------------
    # Public API used by the pipeline/logging configuration
    # ------------------------------------------------------------------
    def get_rich_console(self) -> Any:
        # Return the sink console (writes through the tee → log panel) while the
        # Live display is running.  Fall back to the real console after shutdown.
        if self._stdout_tee is not None and self._sink_console is not None:
            return self._sink_console
        return self.console

    def set_active_prompt(self, message: str) -> None:
        """Show *message* in the footer while waiting for user input."""
        with self._lock:
            self._active_prompt = message
            self._refresh_locked()

    def clear_active_prompt(self) -> None:
        """Restore the normal footer after the prompt has been answered."""
        with self._lock:
            self._active_prompt = None
            self._refresh_locked()

    def set_pipeline_metadata(self, metadata: dict[str, Any]) -> None:
        with self._lock:
            for key, value in metadata.items():
                self.extra_metrics[key] = str(value)
            self._refresh_locked()

    def start_pipeline(
        self,
        name: str,
        total_steps: int,
        step_names: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._view_mode = "modern"
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
            self._full_log_history.clear()
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
        metrics: dict[str, Any] | None = None,
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
            self._append_log("SUCCESS", f"✔ {name} ({self._format_duration(duration)})")
            self._refresh_locked()

    def pipeline_complete(self, success: bool, duration: float) -> None:
        with self._lock:
            self.pipeline_status = "Completed" if success else "Finished"
            self.extra_metrics["elapsed"] = self._format_duration(duration)
            self.current_step_name = "–"
            self._append_log("INFO", f"Pipeline completed in {self._format_duration(duration)}")
            self._pipeline_done = True
            self._stop_ticker_locked()
            self._refresh_locked()

    def pipeline_failed(
        self,
        duration: float,
        error: Exception,
        step_index: int | None,
        step_name: str | None,
    ) -> None:
        with self._lock:
            self.pipeline_status = "Failed"
            self.extra_metrics["elapsed"] = self._format_duration(duration)
            if step_index is not None and self._ensure_index(step_index):
                state = self.step_states[step_index]
                state.status = "failed"
                if step_name:
                    state.name = step_name
            self._append_log("ERROR", f"✖ {type(error).__name__}: {error}")
            self._pipeline_done = True
            self._stop_ticker_locked()
            self._refresh_locked()

    def update_step_progress(
        self,
        index: int,
        completed: float,
        total: float | None = None,
        message: str | None = None,
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

    # ------------------------------------------------------------------
    # Channel-sequential batch lifecycle
    # ------------------------------------------------------------------
    def start_channel_batch(
        self,
        processor_names: list[str],
        channel_names: list[str],
        batch_step_offset: int = 0,
    ) -> None:
        with self._lock:
            self._channel_batch = ChannelBatchState(
                active=True,
                processor_names=list(processor_names),
                channel_names=list(channel_names),
                batch_step_offset=batch_step_offset,
                total_channels=len(channel_names),
                batch_start_time=time.time(),
            )
            self._refresh_locked()

    def channel_started(self, ch_index: int, ch_name: str) -> None:
        with self._lock:
            cb = self._channel_batch
            if not cb.active:
                return
            cb.current_ch_index = ch_index
            cb.current_ch_name = ch_name
            cb.current_proc_index = -1
            cb.current_proc_start = None
            cb.proc_statuses = ["pending"] * len(cb.processor_names)
            cb.proc_durations = [None] * len(cb.processor_names)
            cb.channel_start_time = time.time()
            self._refresh_locked()

    def channel_processor_started(self, ch_index: int, proc_index: int) -> None:
        with self._lock:
            cb = self._channel_batch
            if not cb.active:
                return
            cb.current_proc_index = proc_index
            if 0 <= proc_index < len(cb.proc_statuses):
                cb.proc_statuses[proc_index] = "running"
            cb.current_proc_start = time.time()
            self._refresh_locked()

    def channel_processor_completed(
        self,
        ch_index: int,
        proc_index: int,
        duration: float,
        skipped: bool = False,
    ) -> None:
        with self._lock:
            cb = self._channel_batch
            if not cb.active:
                return
            if 0 <= proc_index < len(cb.proc_statuses):
                cb.proc_statuses[proc_index] = "skipped" if skipped else "done"
            if 0 <= proc_index < len(cb.proc_durations):
                cb.proc_durations[proc_index] = duration
            cb.current_proc_start = None
            self._refresh_locked()

    def channel_completed(self, ch_index: int, duration: float) -> None:
        with self._lock:
            cb = self._channel_batch
            if not cb.active:
                return
            cb.completed_channels += 1
            cb.channel_durations.append(duration)
            self._refresh_locked()

    def end_channel_batch(self) -> None:
        with self._lock:
            self._channel_batch = ChannelBatchState()
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
    def _install_output_tee(self) -> None:
        """Replace stdout/stderr with tees that also feed the Live Logs panel."""
        if self._stdout_tee is not None:
            return
        self._stdout_tee = _LogTee(self._real_stdout, self, level="INFO")
        self._stderr_tee = _LogTee(self._real_stderr, self, level="WARNING")
        sys.stdout = self._stdout_tee  # type: ignore[assignment]
        sys.stderr = self._stderr_tee  # type: ignore[assignment]
        # Create an external-facing Rich Console that writes through the tee so
        # callers using get_rich_console().print() land in the log panel instead
        # of escaping above the Live display.
        #
        # Width budget: each log entry is prefixed with "HH:MM:SS     LEVEL  "
        # (~19 chars) inside the Classic Log panel which has ~4 chars of chrome.
        # The rendered line must therefore fit in (terminal_width - 23) chars or
        # it wraps mid-entry and looks garbled.
        try:
            sink_width = max(40, self.console.width - 23)
            self._sink_console = Console(
                file=self._stdout_tee,  # type: ignore[arg-type]
                highlight=False,
                force_terminal=True,  # emit ANSI color codes through the tee
                width=sink_width,
            )
        except Exception:
            self._sink_console = None

    def _uninstall_output_tee(self) -> None:
        """Restore original stdout/stderr."""
        if self._stdout_tee is not None:
            sys.stdout = self._real_stdout  # type: ignore[assignment]
            self._stdout_tee = None
        if self._stderr_tee is not None:
            sys.stderr = self._real_stderr  # type: ignore[assignment]
            self._stderr_tee = None

    def _tee_append_log(self, level: str, text: str) -> None:
        """Append tee output to log messages and refresh (called from _LogTee.write)."""
        if not text.strip():
            return
        with self._lock:
            self._append_log(level, text)
            self._refresh_locked()

    def _append_log(self, level: str, text: str, timestamp: str | None = None) -> None:
        if timestamp is None:
            timestamp = time.strftime("%H:%M:%S")
        entry = {
            "time": timestamp,
            "level": level.upper(),
            "style": self._level_color(level),
            "text": text.strip(),
        }
        # Full unbounded history for the classic view
        self._full_log_history.append(entry)
        # Rolling window for the modern view
        self.log_messages.append(entry)
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
    # View toggle
    # ------------------------------------------------------------------
    def _toggle_view(self) -> None:
        """Switch between modern dashboard and classic log view (must be called under _lock)."""
        self._view_mode = "classic" if self._view_mode == "modern" else "modern"
        # Pin to bottom when entering classic so the latest logs are visible first
        self._classic_scroll_offset = 0
        self._refresh_locked()

    # ------------------------------------------------------------------
    # Keyboard listener (POSIX only)
    # ------------------------------------------------------------------
    def _start_keyboard_listener(self) -> None:
        if not _TERMIOS_AVAILABLE:
            return
        if not sys.stdin.isatty():
            return
        self._input_stop.clear()
        self._input_thread = threading.Thread(target=self._keyboard_loop, daemon=True, name="facet-kbd")
        self._input_thread.start()

    def _stop_keyboard_listener(self) -> None:
        if self._input_thread is None:
            return
        self._input_stop.set()
        self._input_thread.join(timeout=0.5)
        self._input_thread = None
        self._restore_terminal()

    def _enable_mouse_tracking(self) -> None:
        """Ask the terminal to report scroll-wheel events (SGR mouse protocol)."""
        try:
            # ?1000h = normal mouse tracking, ?1006h = SGR extended coordinates
            self._real_stdout.write("\x1b[?1000h\x1b[?1006h")
            self._real_stdout.flush()
            self._mouse_tracking = True
        except Exception:
            pass

    def _disable_mouse_tracking(self) -> None:
        """Stop the terminal from sending mouse events."""
        if not self._mouse_tracking:
            return
        try:
            self._real_stdout.write("\x1b[?1006l\x1b[?1000l")
            self._real_stdout.flush()
        except Exception:
            pass
        self._mouse_tracking = False

    def _restore_terminal(self) -> None:
        self._disable_mouse_tracking()
        if self._orig_term_settings is not None:
            with contextlib.suppress(Exception):
                _termios.tcsetattr(sys.stdin.fileno(), _termios.TCSADRAIN, self._orig_term_settings)
            self._orig_term_settings = None

    @contextlib.contextmanager
    def suspend_raw_mode(self) -> Generator[None, None, None]:
        """Temporarily restore cooked terminal mode so callers can use input()/readline().

        Pauses the keyboard listener thread, restores the original terminal
        settings for the duration of the ``with`` block, then re-applies raw
        mode when the block exits.  Safe to call from any thread.  Falls back
        to a no-op when the keyboard listener was never started (e.g. stdin is
        not a TTY or termios is unavailable).
        """
        if not _TERMIOS_AVAILABLE or self._orig_term_settings is None or self._input_thread is None:
            yield
            return

        fd = sys.stdin.fileno()
        # Ask the keyboard loop to pause and wait for it to acknowledge.
        self._raw_suspend.set()
        self._raw_paused.wait(timeout=0.5)
        # Discard any raw keypresses that arrived while we were waiting so they
        # don't leak into the caller's input() call.
        with contextlib.suppress(Exception):
            _termios.tcflush(fd, _termios.TCIFLUSH)
        # Stop mouse events while the caller owns stdin.
        self._disable_mouse_tracking()
        # Restore original (cooked) terminal settings.
        with contextlib.suppress(Exception):
            _termios.tcsetattr(fd, _termios.TCSADRAIN, self._orig_term_settings)
        try:
            yield
        finally:
            # Re-apply raw mode settings so the keyboard loop can resume.
            if self._raw_mode_settings is not None:
                with contextlib.suppress(Exception):
                    _termios.tcsetattr(fd, _termios.TCSADRAIN, self._raw_mode_settings)
            self._enable_mouse_tracking()
            self._raw_paused.clear()
            self._raw_suspend.clear()

    # ------------------------------------------------------------------
    # Signal handlers (SIGTERM / SIGHUP) – restore terminal on exit
    # ------------------------------------------------------------------
    def _make_signal_handler(self, sig: int, old_handler: Any):
        def handler(signum: int, frame: Any) -> None:
            self._restore_terminal()
            # Restore the previous handler and re-deliver the signal so the
            # process actually terminates (or the old handler runs).
            try:
                signal.signal(sig, old_handler if old_handler not in (None,) else signal.SIG_DFL)
            except Exception:
                signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)

        return handler

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                old = signal.getsignal(sig)
                signal.signal(sig, self._make_signal_handler(sig, old))
                self._old_signal_handlers[sig] = old
            except (OSError, ValueError):
                # signal.signal() raises ValueError when called from a non-main
                # thread; silently skip in that case.
                pass

    def _uninstall_signal_handlers(self) -> None:
        for sig, old in self._old_signal_handlers.items():
            with contextlib.suppress(Exception):
                signal.signal(sig, old)
        self._old_signal_handlers.clear()

    def _keyboard_loop(self) -> None:
        """Read keypresses in raw mode and dispatch actions."""
        fd = sys.stdin.fileno()
        try:
            self._orig_term_settings = _termios.tcgetattr(fd)
            # Set raw mode for *input* only -- do NOT touch OFLAG/OPOST so that
            # Rich's \n -> \r\n output conversion stays intact.  tty.setraw()
            # clears OPOST which breaks the Live display (lines no longer carry
            # a carriage-return, collapsing everything onto one horizontal line).
            mode = _termios.tcgetattr(fd)
            mode[0] &= ~(  # iflag: disable special input processing
                _termios.BRKINT | _termios.ICRNL | _termios.INPCK | _termios.ISTRIP | _termios.IXON
            )
            # mode[1] (oflag) intentionally left unchanged -- preserves OPOST/ONLCR
            mode[2] = (mode[2] & ~_termios.CSIZE) | _termios.CS8  # cflag: 8-bit chars
            mode[3] &= ~(  # lflag: disable canonical mode, echo, signals
                _termios.ECHO | _termios.ICANON | _termios.IEXTEN | _termios.ISIG
            )
            mode[6][_termios.VMIN] = 1  # type: ignore[index]  # read one byte at a time
            mode[6][_termios.VTIME] = 0  # type: ignore[index]  # no timeout
            _termios.tcsetattr(fd, _termios.TCSADRAIN, mode)
            # Save the raw settings so suspend_raw_mode() can restore them.
            self._raw_mode_settings = copy.deepcopy(_termios.tcgetattr(fd))
        except Exception:
            return

        def _read_byte() -> str | None:
            """Read exactly one byte from the raw FD, bypassing Python's TextIO buffer."""
            try:
                return os.read(fd, 1).decode("utf-8", errors="replace")
            except Exception:
                return None

        def _read_esc_seq() -> str:
            """Read the bytes that follow ESC until the sequence is complete.

            Handles two kinds:
            * Two-char sequences  ESC <X>  where X is any single byte.
            * CSI sequences       ESC [ … <final>  where the body is
              parameter/intermediate bytes (0x20–0x3F) followed by a single
              final byte (0x40–0x7E).

            '[' (0x5B) is the CSI introducer — it is in 0x40–0x7E but is NOT
            a final byte, so we must NOT stop there.  We read it as the first
            character and then switch to "CSI body" mode where we keep reading
            until the actual final byte arrives.

            All reads use os.read() on the raw FD so Python's TextIO buffer
            can never hide bytes from select().
            """
            r, _, _ = _select.select([fd], [], [], 0.05)
            if not r:
                return ""
            ch1 = _read_byte()
            if ch1 is None:
                return ""

            if ch1 != "[":
                # Two-char escape sequence (e.g. ESC O A for SS3 keys) — done.
                return ch1

            # CSI sequence (ESC [): read body until the final byte (0x40–0x7E).
            seq = "["
            for _ in range(63):
                r, _, _ = _select.select([fd], [], [], 0.05)
                if not r:
                    break
                b = _read_byte()
                if b is None:
                    break
                seq += b
                # Parameter/intermediate bytes are 0x20–0x3F; final byte is 0x40–0x7E.
                if len(b) == 1 and 0x40 <= ord(b) <= 0x7E:
                    break
            return seq

        # Rate-limit mouse scroll so touchpad momentum scrolling doesn't fly
        # past every line.  Arrow-key scrolling is not rate-limited.
        _MOUSE_COOLDOWN = 0.06  # seconds — ~12 events/s max for mouse scroll
        _last_mouse_scroll: list[float] = [0.0]

        def _scroll(delta: int, ratelimit: bool = False, page: bool = False) -> None:
            """Adjust classic-view scroll offset by *delta* lines (positive = back).

            Pass ``ratelimit=True`` for mouse-wheel events so touchpad momentum
            scrolling is throttled; arrow-key calls omit the flag.
            Pass ``page=True`` to treat *delta* as a direction sign (±1) and
            scroll by a full viewport height instead.
            """
            if ratelimit:
                now = time.time()
                if now - _last_mouse_scroll[0] < _MOUSE_COOLDOWN:
                    return
                _last_mouse_scroll[0] = now
            with self._lock:
                if self._view_mode != "classic":
                    return
                total = len(self._full_log_history)
                # Viewport height mirrors _render_classic() calculation.
                th = self._terminal_height()
                max_lines = max(1, th - 1 - 2)  # footer=1, panel_chrome=2
                # Maximum useful offset is when the first entry just enters
                # the top of the viewport.  Going further shows nothing new.
                max_offset = max(0, total - max_lines)
                effective_delta = (max_lines if delta > 0 else -max_lines) if page else delta
                self._classic_scroll_offset = max(0, min(self._classic_scroll_offset + effective_delta, max_offset))
                self._refresh_locked()

        self._enable_mouse_tracking()

        try:
            while not self._input_stop.is_set():
                # Cooperate with suspend_raw_mode(): pause here when requested
                # so the terminal can be safely switched to cooked mode.
                if self._raw_suspend.is_set():
                    self._raw_paused.set()
                    while self._raw_suspend.is_set() and not self._input_stop.is_set():
                        time.sleep(0.05)
                    self._raw_paused.clear()
                    continue

                try:
                    ready, _, _ = _select.select([fd], [], [], 0.1)
                except Exception:
                    break
                if not ready:
                    continue

                # Re-check after select in case suspension was requested while
                # we were waiting -- don't read if we're about to hand off stdin.
                if self._raw_suspend.is_set():
                    continue

                ch = _read_byte()
                if ch is None:
                    break

                if ch in ("l", "L"):
                    with self._lock:
                        self._toggle_view()
                elif ch in ("q", "Q"):
                    if self._pipeline_done:
                        self._quit_event.set()
                        break
                    else:
                        os.kill(os.getpid(), signal.SIGINT)
                        break
                elif ch == "\x1b":
                    # We use os.read() directly — all bytes of an escape
                    # sequence may arrive in one OS read, which would put
                    # them in Python's TextIO buffer and hide them from
                    # select().  _read_esc_seq() uses the raw FD throughout.
                    try:
                        seq = _read_esc_seq()
                        if seq == "[A":  # Up arrow
                            _scroll(+3)
                        elif seq == "[B":  # Down arrow
                            _scroll(-3)
                        elif seq == "[1;2A":  # Shift+Up — large step
                            _scroll(+10)
                        elif seq == "[1;2B":  # Shift+Down — large step
                            _scroll(-10)
                        elif seq == "[5~":  # Page Up — full page
                            _scroll(+1, page=True)
                        elif seq == "[6~":  # Page Down — full page
                            _scroll(-1, page=True)
                        elif seq.startswith("[<"):
                            # SGR mouse event: ESC [ < Pb ; Px ; Py M/m
                            # Base button 64=scroll up, 65=scroll down.
                            # Modifier bits are OR-ed into Pb: Shift=+4, Ctrl=+16.
                            #   68/69  = Shift+scroll  → full-page jump
                            #   80/81  = Ctrl+scroll   → full-page jump
                            #   64/65  = plain scroll  → small step
                            last = seq[-1] if seq else ""
                            if last in ("M", "m"):
                                try:
                                    pb = int(seq[2:-1].split(";")[0])
                                    if pb in (68, 80):  # Shift/Ctrl + scroll up
                                        _scroll(+1, ratelimit=True, page=True)
                                    elif pb in (69, 81):  # Shift/Ctrl + scroll down
                                        _scroll(-1, ratelimit=True, page=True)
                                    elif pb == 64:
                                        _scroll(+3, ratelimit=True)
                                    elif pb == 65:
                                        _scroll(-3, ratelimit=True)
                                except (ValueError, IndexError):
                                    pass
                    except Exception:
                        pass
        finally:
            self._restore_terminal()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render(self):  # type: ignore[override]
        if self._view_mode == "classic":
            return self._render_classic()
        return self._render_modern()

    def _render_modern(self):
        if Layout is None:  # pragma: no cover - guard for safety
            return ""
        layout = Layout(name="root")
        header_size, body_size, log_size, footer_size = self._layout_sizes()
        layout.split_column(
            Layout(self._render_header(), name="header", size=header_size),
            Layout(name="body", size=body_size if body_size > 0 else None),
            Layout(self._render_logs_panel(), name="logs", size=log_size if log_size > 0 else None),
            Layout(self._render_footer(), name="footer", size=footer_size),
        )
        body = Layout(name="body")
        body.split_row(
            Layout(self._render_steps_panel(), name="steps", ratio=3),
            Layout(self._render_metrics_panel(), name="metrics", ratio=2),
        )
        layout["body"].update(body)
        return layout

    def _render_classic(self):
        """Full-terminal-height scrolling log view rendered via Rich Live."""
        if Layout is None or Panel is None:  # pragma: no cover
            return ""
        terminal_height = self._terminal_height()
        footer_size = 1
        # Panel chrome: top border (1) + bottom border (1) = 2; title takes border row
        panel_chrome = 2
        max_lines = max(1, terminal_height - footer_size - panel_chrome)

        history = self._full_log_history
        total = len(history)

        # Apply scroll offset: offset=0 means pinned to bottom.
        if self._classic_scroll_offset > 0:
            # Floor end_idx so it never drops below the viewport height —
            # this prevents rows disappearing when scrolled all the way up.
            end_idx = max(min(max_lines, total), total - self._classic_scroll_offset)
            start_idx = max(0, end_idx - max_lines)
            visible = history[start_idx:end_idx]
        else:
            visible = history[-max_lines:] if total > max_lines else history

        if not visible:
            body: Any = Align.left(Text("No log entries yet", style="dim"))
        else:
            rows = []
            for entry in visible:
                row = Text()
                row.append(entry["time"], style="dim")
                row.append("  ")
                row.append(f"{entry['level']:>7}", style=entry["style"])
                row.append("  ")
                row.append_text(Text.from_ansi(entry["text"]))
                rows.append(row)
            body = Align.left(Group(*rows))

        # Title: show scroll position when not at the bottom
        if self._classic_scroll_offset > 0 and total > max_lines:
            end_shown = total - self._classic_scroll_offset
            title = f"Classic Log  ({end_shown}/{total} — scrolled up)"
        else:
            title = f"Classic Log  ({total} entries)"

        log_panel = Panel(
            body,
            title=title,
            border_style="bright_black",
            box=box.ROUNDED,
        )

        layout = Layout(name="root")
        layout.split_column(
            Layout(log_panel, name="logs", size=terminal_height - footer_size),
            Layout(self._render_footer(), name="footer", size=footer_size),
        )
        return layout

    def _render_footer(self) -> Text:
        """One-line keybinding hint bar shown in both views."""
        # While a prompt is active, replace the normal hints with the prompt text.
        if self._active_prompt is not None:
            t = Text(justify="center")
            t.append(" ⏎ ", style="bold yellow")
            t.append(self._active_prompt, style="bold yellow")
            return t

        t = Text(justify="center")
        t.append(" [", style="dim")
        t.append("l", style="bold cyan")
        t.append("] toggle view    [", style="dim")
        t.append("q", style="bold red")
        q_label = "] close " if self._pipeline_done else "] quit "
        t.append(q_label, style="dim")
        if self._view_mode == "classic":
            t.append("    [", style="dim")
            t.append("↑↓", style="bold cyan")
            t.append("] scroll  [", style="dim")
            t.append("Shift+↑↓", style="bold cyan")
            t.append(" / ", style="dim")
            t.append("PgUp/Dn", style="bold cyan")
            t.append("] fast    ◀ classic view", style="dim yellow")
        return t

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
        if self._channel_batch.active:
            cb = self._channel_batch
            text.append(f"  •  Channel {cb.completed_channels + 1}/{cb.total_channels}", style="bold cyan")
        elapsed = self._elapsed_seconds()
        if elapsed > 0:
            text.append(f"  •  {self._format_duration(elapsed)} elapsed", style="dim")
        if self._pipeline_done:
            text.append("    press ", style="dim")
            text.append("q", style="bold red")
            text.append(" to close", style="dim")
        return Panel(text, border_style="bright_black", padding=(0, 1))

    def _render_steps_panel(self) -> Panel:
        if self._channel_batch.active:
            return self._render_channel_batch_panel()

        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column("Processor", style="bold", ratio=3)
        table.add_column("Status", style="white", width=14, justify="left")
        table.add_column("Progress", style="white", width=22)
        table.add_column("Duration", style="white", width=8, justify="left")

        if not self.step_states:
            table.add_row("Waiting for pipeline", "[dim]idle[/]", "", "–")
        else:
            for step in self.step_states:
                icon, style = self._status_badge(step.status)
                duration = (
                    self._format_duration(step.duration)
                    if step.duration
                    else ("…" if step.status == "running" else "–")
                )
                display_name = step.name or f"Step {step.index + 1}"
                table.add_row(
                    display_name,
                    f"[{style}]{icon} {step.status.capitalize()}[/]",
                    self._step_progress(step),
                    duration,
                )

        return Panel(table, title="Processors", border_style="bright_black", box=box.ROUNDED)

    def _render_channel_batch_panel(self) -> Panel:
        """Render the steps panel during a channel-sequential batch."""
        cb = self._channel_batch
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column("Processor", style="bold", ratio=3)
        table.add_column("Status", style="white", width=18, justify="left")
        table.add_column("Duration", style="white", width=10, justify="left")

        for i, proc_name in enumerate(cb.processor_names):
            status = cb.proc_statuses[i] if i < len(cb.proc_statuses) else "pending"
            dur = cb.proc_durations[i] if i < len(cb.proc_durations) else None

            if status == "running":
                icon, sty = "▶", "cyan"
                if cb.current_proc_start:
                    elapsed = time.time() - cb.current_proc_start
                    dur_str = f"{self._format_duration(elapsed)}…"
                else:
                    dur_str = "…"
            elif status == "done":
                icon, sty = "✔", "green"
                dur_str = self._format_duration(dur) if dur is not None else ""
            elif status == "skipped":
                icon, sty = "⊘", "yellow"
                dur_str = "run_once"
            else:
                icon, sty = "●", "grey62"
                dur_str = "–"

            table.add_row(
                proc_name,
                f"[{sty}]{icon} {status.capitalize()}[/]",
                dur_str,
            )

        ch_num = cb.completed_channels + 1
        ch_display = cb.current_ch_name or "–"
        title = f"Channel {ch_num}/{cb.total_channels}: {ch_display}"
        return Panel(table, title=title, border_style="cyan", box=box.ROUNDED)

    def _render_metrics_panel(self) -> Panel:
        if self._channel_batch.active:
            return self._render_channel_metrics_panel()

        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column("Metric", style="bold", width=16, justify="left")
        table.add_column("Value", style="white", justify="left")

        table.add_row("Progress", self._progress_bar())
        table.add_row("Completed", f"{self.completed_steps}/{max(self.total_steps, 1)}")
        table.add_row("Current", self.current_step_name)
        last = (
            f"{self.last_step_name} ({self._format_duration(self.last_step_duration)})"
            if self.last_step_duration
            else self.last_step_name
        )
        table.add_row("Last", last)
        table.add_row("Elapsed", self._format_duration(self._elapsed_seconds()))

        diagnostics, reported = self._split_extra_metrics()
        for key, value in diagnostics:
            table.add_row(key.replace("_", " ").title(), value)

        if reported:
            table.add_row("", "")
            table.add_row(Text("Reported", style="bold yellow underline"), "")
            for key, value in reported:
                table.add_row(key.replace("_", " ").title(), value)

        return Panel(table, title="Metrics", border_style="bright_black", box=box.ROUNDED)

    def _render_channel_metrics_panel(self) -> Panel:
        """Render the metrics panel during a channel-sequential batch."""
        cb = self._channel_batch
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column("Metric", style="bold", width=16, justify="left")
        table.add_column("Value", style="white", justify="left")

        ratio = cb.completed_channels / max(cb.total_channels, 1)
        bar = self._inline_bar(ratio, width=14)
        table.add_row("Channels", f"{cb.completed_channels}/{cb.total_channels}")
        table.add_row("Progress", f"{bar} {int(ratio * 100):>3d}%")
        table.add_row("Current", cb.current_ch_name or "–")

        if cb.channel_durations:
            last_idx = cb.completed_channels - 1
            last_name = cb.channel_names[last_idx] if 0 <= last_idx < len(cb.channel_names) else "–"
            table.add_row("Last Ch", f"{last_name} ({self._format_duration(cb.channel_durations[-1])})")

            avg = sum(cb.channel_durations) / len(cb.channel_durations)
            remaining = max(cb.total_channels - cb.completed_channels, 0)
            eta = avg * remaining
            table.add_row("Avg / Ch", self._format_duration(avg))
            table.add_row("Est. Remain", f"~{self._format_duration(eta)}")
        else:
            table.add_row("Last Ch", "–")

        if cb.batch_start_time:
            batch_elapsed = time.time() - cb.batch_start_time
            table.add_row("Batch Elapsed", self._format_duration(batch_elapsed))

        table.add_row("Total Elapsed", self._format_duration(self._elapsed_seconds()))

        return Panel(table, title="Channel Progress", border_style="bright_black", box=box.ROUNDED)

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
                row.append_text(Text.from_ansi(entry["text"]))
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
    def _format_duration(seconds: float) -> str:
        """Format seconds as days, hours, minutes, and seconds."""
        if seconds < 0:
            return "0.00s"
        s = int(round(seconds))
        d, s = divmod(s, 86400)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        parts = []
        if d:
            parts.append(f"{d}d")
        if h:
            parts.append(f"{h}h")
        if m:
            parts.append(f"{m}m")
        # When only seconds, show two decimals for finer granularity
        if d or h or m:
            parts.append(f"{s}s")
        else:
            parts.append(f"{seconds:.2f}s")
        return " ".join(parts)

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

    def _layout_sizes(self) -> tuple[int, int, int, int]:
        header_size = 3
        footer_size = 1
        min_log_size = 6
        total_height = self._terminal_height()
        available = max(total_height - header_size - footer_size, 0)
        body_needed = self._body_panel_height_needed()

        if available <= 0:
            return header_size, 0, 0, footer_size

        if available >= body_needed + min_log_size:
            body_size = body_needed
            log_size = available - body_size
        else:
            body_size = min(body_needed, available)
            log_size = available - body_size

        return header_size, body_size, max(log_size, 0), footer_size

    def _body_panel_height_needed(self) -> int:
        processors_height = self._processors_panel_height_needed()
        metrics_height = self._metrics_panel_height_needed()
        return max(processors_height, metrics_height)

    def _processors_panel_height_needed(self) -> int:
        if self._channel_batch.active:
            rows = max(len(self._channel_batch.processor_names), 1)
        else:
            rows = len(self.step_states) if self.step_states else 1
        panel_chrome = 4
        return rows + panel_chrome

    def _metrics_panel_height_needed(self) -> int:
        if self._channel_batch.active:
            base_rows = 8
            panel_chrome = 4
            return base_rows + panel_chrome
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
        _, _, log_size, _ = self._layout_sizes()
        if log_size <= 0:
            return 0
        panel_chrome = 4
        return max(1, log_size - panel_chrome)
