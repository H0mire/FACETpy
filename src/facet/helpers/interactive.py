"""
Interactive Helpers Module

Processors that facilitate interactive pipeline steps such as awaiting user
confirmation before continuing execution.

Author: FACETpy Team
Date: 2025-01-12
"""

import os
import sys
import time
from typing import Optional

from loguru import logger

from ..console import get_console, suspend_raw_mode
from ..core import Processor, ProcessingContext, register_processor


@register_processor
class WaitForConfirmation(Processor):
    """
    Pause pipeline execution until the user confirms continuation.

    Designed for iterative, notebook-driven, or CLI debugging workflows where
    manual inspection is required between processing stages. When interactive
    input is unavailable, the processor automatically continues to avoid
    blocking headless runs.
    """

    name = "wait_for_confirmation"
    description = "Pause pipeline until user confirmation."
    modifies_raw = False

    def __init__(
        self,
        message: str = "Press Enter to continue...",
        auto_continue: bool = False,
        timeout: Optional[float] = None,
        continue_on_timeout: bool = True
    ):
        """
        Initialize the confirmation step.

        Args:
            message: Prompt presented to the user.
            auto_continue: Skip the pause entirely when True.
            timeout: Optional timeout in seconds before continuing automatically.
            continue_on_timeout: Whether to resume automatically after timeout
                expires. If False, raises a TimeoutError instead.
        """
        self.message = message
        self.auto_continue = auto_continue
        self.timeout = timeout
        self.continue_on_timeout = continue_on_timeout
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Wait for the user to confirm or continue automatically."""
        if self.auto_continue:
            logger.info("Auto-continue enabled; skipping confirmation step.")
            return context

        if not sys.stdin or not sys.stdin.isatty():
            logger.warning("Standard input is not interactive; continuing automatically.")
            return context

        # Print the (optionally Rich-markup) message through the log panel so it
        # lands inside the live display rather than escaping above it.
        self._print_message()

        # Derive a short single-line footer hint — strip any Rich markup tags so
        # they don't appear literally in the footer text.
        try:
            from rich.text import Text as _RichText
            _strip = lambda s: _RichText.from_markup(s).plain
        except Exception:
            import re as _re
            _strip = lambda s: _re.sub(r"\[/?[^\]]*\]", "", s)
        footer_hint = next(
            (_strip(l.strip()) for l in self.message.split("\n") if l.strip()),
            "Press Enter to continue...",
        )
        console = get_console()
        console.set_active_prompt(footer_hint)
        try:
            # Suspend raw terminal mode while waiting so input()/readline() work
            # correctly even when the ModernConsole keyboard listener is active.
            with suspend_raw_mode():
                try:
                    if self.timeout is None:
                        input("")
                    else:
                        self._prompt_with_timeout()
                except (EOFError, KeyboardInterrupt):
                    logger.info("User aborted confirmation step; continuing execution.")
                except TimeoutError as exc:
                    logger.warning(str(exc))
        finally:
            console.clear_active_prompt()
        return context

    def _print_message(self) -> None:
        """Render the message through the Rich console, auto-colouring plain text."""
        rich_console = get_console().get_rich_console()
        # If the message contains no Rich markup tags, apply a default colour scheme:
        # the first non-empty line is rendered bold+yellow, remaining lines dim.
        has_markup = "[" in self.message
        first = True
        for line in self.message.split("\n"):
            line = line.strip()
            if not line:
                continue
            if has_markup:
                rich_console.print(line)
            elif first:
                rich_console.print(f"[bold yellow]{line}[/bold yellow]")
            else:
                rich_console.print(f"[dim]{line}[/dim]")
            first = False

    def _prompt_with_timeout(self) -> None:
        """Wait for user confirmation with an optional timeout."""
        logger.info(
            "Waiting for user confirmation (timeout=%.1fs)...",
            self.timeout
        )
        start_time = time.time()

        if os.name == "nt":
            self._wait_windows(start_time)
        else:
            self._wait_posix(start_time)

    def _wait_windows(self, start_time: float) -> None:
        """Handle confirmation on Windows platforms."""
        import msvcrt

        while True:
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ("\n", "\r"):
                    return
            if self.timeout is not None and (time.time() - start_time) > self.timeout:
                self._handle_timeout()
                return
            time.sleep(0.05)

    def _wait_posix(self, start_time: float) -> None:
        """Handle confirmation on POSIX platforms using select."""
        import select

        ready, _, _ = select.select([sys.stdin], [], [], self.timeout)
        if ready:
            sys.stdin.readline()
        else:
            self._handle_timeout()

    def _handle_timeout(self) -> None:
        """Handle timeout conditions according to configuration."""
        if self.continue_on_timeout:
            logger.warning("Confirmation timeout reached; continuing automatically.")
        else:
            raise TimeoutError("Confirmation timeout reached and continue_on_timeout=False.")
