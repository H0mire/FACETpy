from __future__ import annotations

import signal
import threading
from unittest.mock import patch

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
