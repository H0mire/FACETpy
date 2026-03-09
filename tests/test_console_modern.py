from __future__ import annotations

import signal
import threading
import time
from unittest.mock import Mock, patch

from facet.console.modern import ModernConsole


def _make_console(*, pipeline_done: bool) -> ModernConsole:
    console = ModernConsole.__new__(ModernConsole)
    console._pipeline_done = pipeline_done
    console._quit_event = threading.Event()
    return console


def test_handle_quit_key_interrupts_when_pipeline_is_running() -> None:
    console = _make_console(pipeline_done=False)
    with (
        patch("facet.console.modern.os.getpid", return_value=4242),
        patch("facet.console.modern.os.kill") as kill,
    ):
        should_close = console._handle_quit_key()

    assert should_close is False
    assert console._quit_event.is_set()
    kill.assert_called_once_with(4242, signal.SIGINT)


def test_handle_quit_key_closes_when_pipeline_finished() -> None:
    console = _make_console(pipeline_done=True)
    with patch("facet.console.modern.os.kill") as kill:
        should_close = console._handle_quit_key()

    assert should_close is True
    assert console._quit_event.is_set()
    kill.assert_not_called()


def test_resize_signal_handler_sets_refresh_pending_and_calls_old_handler() -> None:
    console = ModernConsole.__new__(ModernConsole)
    console._resize_refresh_pending = threading.Event()
    old_handler = Mock()

    handler = console._make_resize_signal_handler(old_handler)
    handler(signal.SIGTERM, None)

    assert console._resize_refresh_pending.is_set()
    old_handler.assert_called_once()


def test_resize_loop_refreshes_when_pending_event_is_set() -> None:
    console = ModernConsole.__new__(ModernConsole)
    console._resize_stop = threading.Event()
    console._resize_refresh_pending = threading.Event()
    console._lock = threading.Lock()
    console._rebuild_sink_console = Mock()
    console._trim_logs = Mock()
    console._refresh_locked = Mock()

    thread = threading.Thread(target=console._resize_loop, daemon=True)
    thread.start()
    console._resize_refresh_pending.set()

    for _ in range(10):
        if console._refresh_locked.called:
            break
        time.sleep(0.02)

    console._resize_stop.set()
    console._resize_refresh_pending.set()
    thread.join(timeout=0.5)

    assert console._refresh_locked.called
