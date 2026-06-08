"""
Interactive Helpers Module

Processors that facilitate interactive pipeline steps such as awaiting user
confirmation before continuing execution.

Author: FACETpy Team
Date: 2025-01-12
"""

import contextlib
import json
import os
import sys
import time

import mne
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

from ..console import get_console, suspend_raw_mode
from ..core import (
    ProcessingContext,
    Processor,
    ProcessorError,
    ProcessorValidationError,
    register_processor,
)
from .plotting import show_matplotlib_figure


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
        timeout: float | None = None,
        continue_on_timeout: bool = True,
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

            def _strip(s):
                return _RichText.from_markup(s).plain
        except Exception:
            import re as _re

            def _strip(s):
                return _re.sub(r"\[/?[^\]]*\]", "", s)

        footer_hint = next(
            (_strip(line.strip()) for line in self.message.split("\n") if line.strip()),
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
            if rich_console is None:
                logger.info(line)
            elif has_markup:
                rich_console.print(line)
            elif first:
                rich_console.print(f"[bold yellow]{line}[/bold yellow]")
            else:
                rich_console.print(f"[dim]{line}[/dim]")
            first = False

    def _prompt_with_timeout(self) -> None:
        """Wait for user confirmation with an optional timeout."""
        logger.info("Waiting for user confirmation (timeout=%.1fs)...", self.timeout)
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


@register_processor
class ArtifactOffsetFinder(Processor):
    """Interactively determine the artifact-to-trigger offset.

    Displays a matplotlib plot of overlaid EEG epochs centred on the trigger
    positions.  A slider and click-to-set interface let the user visually
    align an artifact window with the data.  The confirmed offset is written
    to ``context.metadata.artifact_to_trigger_offset``.

    The plot shows multiple overlaid epochs (mean-subtracted) so the
    repeating artifact pattern is clearly visible.  A blue dashed line marks
    the trigger; a red line and shaded region mark the current artifact
    window.  Drag the slider or left-click on the plot to reposition the
    offset, then press **Confirm**.

    In addition to the offset, the artifact **length** (the width of the
    shaded window) can be adjusted with a second slider; the confirmed length
    is written to ``context.metadata.artifact_length``.

    Parameters
    ----------
    channel : str | int | None
        Channel to display.  Name (str), index (int), or ``None`` for the
        first EEG channel.
    n_epochs : int
        Number of artifact epochs to overlay (default: 5).
    initial_offset : float | None
        Starting offset in seconds.  When ``None`` (default), the current
        context value is used.
    artifact_length : int | None
        Starting artifact length in samples for the length slider.  When
        ``None`` (default), the current context value is used.  This also
        serves as a programmatic override: the confirmed value is always
        written back to the context.
    """

    name = "artifact_offset_finder"
    description = "Interactively find artifact-to-trigger offset and length"
    version = "1.1.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        channel: str | int | None = None,
        n_epochs: int = 5,
        initial_offset: float | None = None,
        artifact_length: int | None = None,
    ) -> None:
        self.channel = channel
        self.n_epochs = n_epochs
        self.initial_offset = initial_offset
        self.artifact_length = artifact_length
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None and self.artifact_length is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector first or pass artifact_length."
            )
        n_triggers = len(context.get_triggers())
        if n_triggers < 2:
            raise ProcessorValidationError(f"Need at least 2 triggers to determine offset, got {n_triggers}.")
        if self.n_epochs < 1:
            raise ProcessorValidationError(f"n_epochs must be >= 1, got {self.n_epochs}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        sfreq = context.get_sfreq()
        base_length = self.artifact_length if self.artifact_length is not None else context.get_artifact_length()
        current_offset = context.metadata.artifact_to_trigger_offset

        # --- LOG ---
        logger.info(
            "Opening artifact offset finder (current offset={:.4f} s, artifact length={} samples)",
            current_offset,
            int(base_length),
        )

        # --- COMPUTE ---
        ch_idx = self._resolve_channel(raw)
        base_duration = base_length / sfreq

        # Slider ranges: offset within +/- half a period; length 0.25x .. 2x the
        # starting length. Extract a window wide enough that the widest possible
        # shade (max offset + max length) always fits on screen.
        padding = base_duration * 0.25
        max_offset = base_duration * 0.5
        # Generous lower bound: let the user shrink the window down to a single
        # sample. (Only the upper bound matters for how wide the data is
        # extracted, so a tiny minimum is free.)
        min_length = 1.0 / sfreq
        max_length = base_duration * 2.0
        pre_span = max_offset + padding
        post_span = max_offset + max_length + padding

        epochs_data, time_axis = self._extract_epochs(
            raw._data[ch_idx],
            triggers,
            sfreq,
            pre_span,
            post_span,
        )

        if self.initial_offset is not None:
            start_offset = self.initial_offset
        else:
            auto_offset = self._auto_detect_offset(
                epochs_data,
                time_axis,
                sfreq,
                base_duration,
            )
            if auto_offset is not None:
                start_offset = auto_offset
                logger.info(
                    "Auto-detected artifact onset at {:.4f} s ({:.2f} ms)",
                    auto_offset,
                    auto_offset * 1000,
                )
            else:
                start_offset = current_offset
                logger.debug("Auto-detection failed; using current offset")

        chosen_offset, chosen_duration = self._show_interactive_plot(
            epochs_data,
            time_axis,
            raw.ch_names[ch_idx],
            sfreq,
            start_offset,
            base_duration,
            min_length,
            max_length,
            max_offset,
        )
        chosen_length = max(1, int(round(chosen_duration * sfreq)))

        # --- BUILD RESULT ---
        logger.info(
            "Offset set to {:.6f} s ({:.3f} ms); artifact length set to {} samples ({:.3f} ms)",
            chosen_offset,
            chosen_offset * 1000,
            chosen_length,
            chosen_duration * 1000,
        )
        new_metadata = context.metadata.copy()
        new_metadata.artifact_to_trigger_offset = chosen_offset
        new_metadata.artifact_length = chosen_length

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    # -----------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------

    def _resolve_channel(self, raw: mne.io.Raw) -> int:
        """Determine the channel index to display."""
        if self.channel is None:
            eeg_channels = mne.pick_types(
                raw.info,
                eeg=True,
                exclude="bads",
            )
            if len(eeg_channels) == 0:
                raise ProcessorError("No EEG channels found in raw data.")
            return int(eeg_channels[0])

        if isinstance(self.channel, int):
            return self.channel
        return raw.ch_names.index(self.channel)

    def _extract_epochs(
        self,
        channel_data: np.ndarray,
        triggers: np.ndarray,
        sfreq: float,
        pre_span: float,
        post_span: float,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Extract mean-subtracted epochs centred on triggers.

        Parameters
        ----------
        pre_span : float
            Seconds of data to include before each trigger.
        post_span : float
            Seconds of data to include after each trigger. Sized so the widest
            offset/length slider combination still fits inside the window.

        Returns
        -------
        epochs : list of np.ndarray
            Epoch arrays, one per trigger shown.
        time_axis : np.ndarray
            Time in seconds relative to trigger.
        """
        n_pre = int(pre_span * sfreq)
        n_post = int(post_span * sfreq)
        n_total = n_pre + n_post
        time_axis = (np.arange(n_total) - n_pre) / sfreq

        n_show = min(self.n_epochs, len(triggers) - 1)
        epochs: list[np.ndarray] = []
        for i in range(n_show):
            start = int(triggers[i]) - n_pre
            end = start + n_total
            if start < 0 or end > len(channel_data):
                continue
            epoch = channel_data[start:end].copy()
            epoch -= np.mean(epoch)
            epochs.append(epoch)

        if not epochs:
            raise ProcessorError("Could not extract any epochs for display.")
        return epochs, time_axis

    def _auto_detect_offset(
        self,
        epochs_data: list[np.ndarray],
        time_axis: np.ndarray,
        sfreq: float,
        artifact_duration: float,
    ) -> float | None:
        """Estimate the artifact onset from the averaged epoch shape.

        Averages overlaid epochs to enhance the repeating artifact and
        suppress random EEG.  Because gradient artifacts are contiguous
        (each period immediately follows the previous one), the boundary
        between two periods shows up as a *minimum* in the smoothed
        amplitude envelope.  This method finds the deepest such minimum
        in a window around the trigger.

        Parameters
        ----------
        epochs_data : list of np.ndarray
            Mean-subtracted epochs extracted by :meth:`_extract_epochs`.
        time_axis : np.ndarray
            Time in seconds relative to the trigger.
        sfreq : float
            Sampling frequency in Hz.
        artifact_duration : float
            Duration of one artifact period in seconds.

        Returns
        -------
        float or None
            Estimated offset in seconds relative to the trigger, or
            ``None`` if detection fails.
        """
        if len(epochs_data) < 2:
            return None

        mean_epoch = np.mean(epochs_data, axis=0)

        # Smoothed amplitude envelope — kernel spans ~1 % of the
        # artifact period, enough to iron out sample-level noise
        # without smearing the inter-artifact dip.
        envelope = np.abs(mean_epoch)
        kernel_size = max(5, int(sfreq * artifact_duration * 0.01))
        kernel = np.ones(kernel_size) / kernel_size
        smooth_env = np.convolve(envelope, kernel, mode="same")

        # Search for the amplitude minimum near the trigger (±20 % of
        # artifact duration).  The dip between consecutive artifact
        # periods marks the epoch boundary.
        margin = artifact_duration * 0.2
        search_mask = (time_axis >= -margin) & (time_axis <= margin)
        if not np.any(search_mask):
            return None

        search_indices = np.where(search_mask)[0]
        min_idx = search_indices[np.argmin(smooth_env[search_indices])]
        return float(time_axis[min_idx])

    def _setup_plot_axes(
        self,
        epochs_data: list[np.ndarray],
        time_axis: np.ndarray,
        ch_name: str,
        offset: float,
        artifact_duration: float,
    ) -> tuple:
        """Create the figure with epoch overlay and offset markers.

        Returns
        -------
        tuple
            ``(fig, ax, offset_line, offset_text)`` — the figure, axes,
            moveable offset line, and offset label.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.28)

        for epoch in epochs_data:
            ax.plot(time_axis, epoch, alpha=0.5, linewidth=0.7)

        ax.axvline(0, color="blue", ls="--", lw=1.5, label="Trigger")
        offset_line = ax.axvline(
            offset,
            color="red",
            lw=2,
            label="Artifact start",
        )

        offset_text = ax.text(
            0.02,
            0.95,
            f"Offset: {offset * 1000:.2f} ms",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_xlabel("Time relative to trigger (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Artifact Offset Finder \u2014 {ch_name}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        return fig, ax, offset_line, offset_text

    @staticmethod
    def _length_slider_to_seconds(slider_val: float, sfreq: float) -> float:
        """Map a log-space length-slider position to a sample-quantised length.

        The length slider operates in ``log10(seconds)`` space so that a fixed
        amount of slider travel always changes the window by a fixed *ratio*
        rather than a fixed absolute amount. Near the small end this means a
        given drag adjusts the window by only a few samples — giving fine
        control exactly where the window is tiny — while the same slider still
        spans the full range. The result is snapped to the nearest whole sample
        (never below one sample).

        Parameters
        ----------
        slider_val : float
            Raw slider position in ``log10(seconds)``.
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        float
            Sample-quantised length in seconds.
        """
        length = 10.0**slider_val
        return max(1, round(length * sfreq)) / sfreq

    @staticmethod
    def _compute_view_xlim(
        offset: float,
        length: float,
        t_lo: float,
        t_hi: float,
        sfreq: float,
    ) -> tuple[float, float]:
        """X-axis limits that frame the current artifact window.

        The view widens/narrows with the length slider (the margin scales with
        the window length) while always keeping both the trigger (``t=0``) and
        the full artifact window visible. Clamped to the extracted data extent
        ``[t_lo, t_hi]`` so the view never runs past the available samples.

        Parameters
        ----------
        offset : float
            Current artifact-start offset in seconds (window left edge).
        length : float
            Current artifact length in seconds (window width).
        t_lo, t_hi : float
            First / last time of the extracted epoch window, in seconds.
        sfreq : float
            Sampling frequency in Hz (sets the minimum margin).

        Returns
        -------
        tuple of (float, float)
            ``(xlo, xhi)`` axis limits in seconds.
        """
        margin = max(0.3 * length, 1.0 / sfreq)
        lo = max(min(offset, 0.0) - margin, t_lo)
        hi = min(max(offset + length, 0.0) + margin, t_hi)
        return lo, hi

    def _show_interactive_plot(
        self,
        epochs_data: list[np.ndarray],
        time_axis: np.ndarray,
        ch_name: str,
        sfreq: float,
        start_offset: float,
        start_duration: float,
        min_length: float,
        max_length: float,
        max_offset: float,
    ) -> tuple[float, float]:
        """Display the interactive finder and return the chosen offset and length.

        Adds an offset slider, an artifact-length slider, a click-to-set
        handler, and confirm/cancel buttons on top of the axes created by
        :meth:`_setup_plot_axes`.

        Returns
        -------
        tuple of (float, float)
            ``(offset_seconds, length_seconds)``.
        """
        fig, ax, offset_line, offset_text = self._setup_plot_axes(
            epochs_data,
            time_axis,
            ch_name,
            start_offset,
            start_duration,
        )
        # Make room for the two sliders + button row.
        fig.subplots_adjust(bottom=0.34)
        state: dict = {
            "offset": start_offset,
            "length": start_duration,
            "confirmed": False,
            "shade": None,
            "zoom": True,
        }

        def _draw_shade() -> None:
            if state["shade"] is not None:
                state["shade"].remove()
            state["shade"] = ax.axvspan(
                state["offset"],
                state["offset"] + state["length"],
                alpha=0.12,
                color="red",
                label="_nolegend_",
            )

        def _refresh_label() -> None:
            offset_text.set_text(
                f"Offset: {state['offset'] * 1000:.2f} ms\nLength: {state['length'] * 1000:.2f} ms"
            )

        t_lo, t_hi = float(time_axis[0]), float(time_axis[-1])

        def _rescale_y(mask: np.ndarray, pad_frac: float) -> None:
            if not np.any(mask):
                return
            vals = np.concatenate([epoch[mask] for epoch in epochs_data])
            ymin, ymax = float(np.min(vals)), float(np.max(vals))
            if ymax > ymin:
                pad = pad_frac * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)

        def _update_view() -> None:
            """Zoom the time axis (and amplitude) to the current artifact window."""
            xlo, xhi = self._compute_view_xlim(state["offset"], state["length"], t_lo, t_hi, sfreq)
            if xhi <= xlo:
                return
            ax.set_xlim(xlo, xhi)
            # Rescale amplitude to the data now visible, otherwise zooming the
            # time axis is pointless — the wide initial y-range (set from
            # neighbouring artifacts) would flatten the zoomed window.
            _rescale_y((time_axis >= xlo) & (time_axis <= xhi), pad_frac=0.1)

        _draw_shade()
        _refresh_label()
        _update_view()

        # --- Offset slider ---
        offset_ax = fig.add_axes([0.15, 0.18, 0.65, 0.04])
        offset_slider = Slider(
            offset_ax,
            "Offset (s)",
            -max_offset,
            max_offset,
            valinit=min(max(start_offset, -max_offset), max_offset),
            valstep=1.0 / sfreq,
        )

        def _on_offset_changed(val: float) -> None:
            state["offset"] = val
            offset_line.set_xdata([val, val])
            _draw_shade()
            _refresh_label()
            if state["zoom"]:
                _update_view()
            fig.canvas.draw_idle()

        offset_slider.on_changed(_on_offset_changed)

        # --- Artifact-length slider (logarithmic: fine control at small sizes) ---
        length_ax = fig.add_axes([0.15, 0.11, 0.65, 0.04])
        start_clamped = min(max(start_duration, min_length), max_length)
        length_slider = Slider(
            length_ax,
            "Length (s)",
            float(np.log10(min_length)),
            float(np.log10(max_length)),
            valinit=float(np.log10(start_clamped)),
        )
        # Slider value lives in log10(seconds); show the real length instead.
        length_slider.valtext.set_text(f"{start_clamped:.4f}")

        def _on_length_changed(log_val: float) -> None:
            length = self._length_slider_to_seconds(log_val, sfreq)
            state["length"] = length
            length_slider.valtext.set_text(f"{length:.4f}")
            _draw_shade()
            _refresh_label()
            if state["zoom"]:
                _update_view()
            fig.canvas.draw_idle()

        length_slider.on_changed(_on_length_changed)

        # --- Click-to-set offset ---
        def _on_click(event) -> None:
            if event.inaxes == ax and event.button == 1:
                offset_slider.set_val(event.xdata)

        fig.canvas.mpl_connect("button_press_event", _on_click)

        # --- Confirm / Cancel buttons ---
        confirm_ax = fig.add_axes([0.68, 0.03, 0.12, 0.05])
        confirm_btn = Button(confirm_ax, "Confirm")

        def _close_fig() -> None:
            """Destroy the native window while the event loop is active."""
            with contextlib.suppress(Exception):
                fig.canvas.manager.destroy()
            plt.close(fig)

        def _on_confirm(_) -> None:
            state["confirmed"] = True
            _close_fig()

        confirm_btn.on_clicked(_on_confirm)

        cancel_ax = fig.add_axes([0.82, 0.03, 0.12, 0.05])
        cancel_btn = Button(cancel_ax, "Cancel")
        cancel_btn.on_clicked(lambda _: _close_fig())

        # --- Auto-zoom toggle ---
        zoom_ax = fig.add_axes([0.02, 0.02, 0.13, 0.07])
        zoom_check = CheckButtons(zoom_ax, ["Auto-zoom"], [state["zoom"]])

        def _on_zoom_toggled(_label: str) -> None:
            # Toggling only controls whether *future* slider moves auto-zoom;
            # it must NOT change the current view (manual pan/zoom is kept).
            try:
                state["zoom"] = bool(zoom_check.get_status()[0])
            except AttributeError:  # pragma: no cover - older matplotlib
                state["zoom"] = not state["zoom"]
            fig.canvas.draw_idle()

        zoom_check.on_clicked(_on_zoom_toggled)

        console = get_console()
        console.set_active_prompt("Adjust offset & length in plot window, then click Confirm")
        try:
            with suspend_raw_mode():
                show_matplotlib_figure(fig)
        finally:
            plt.close(fig)
            console.clear_active_prompt()

        if not state["confirmed"]:
            logger.info("Selection cancelled; keeping current offset and artifact length")
            return start_offset, start_duration

        return state["offset"], state["length"]


@register_processor
class TriggerEditor(Processor):
    """Comprehensive interactive trigger editor.

    A superset of :class:`ArtifactOffsetFinder`. On top of overlaying artifact
    epochs and setting the artifact-to-trigger offset, it lets the user:

    * adjust the **offset** (artifact start) and the **artifact length**,
    * subdivide each trigger's artifact field into ``N`` **slice triggers**
      (highlighted as vertical lines inside the field),
    * edit everything either via **text fields** or by **dragging** — the whole
      window moves the offset, the left/right **edges** change the length,
    * **zoom** the time axis with the mouse scroll wheel,
    * **save / load** the current settings (offset, length, slice count) to a
      small JSON file.

    On confirm:

    * with ``slice count == 1``: the offset and artifact length are written to
      the context (triggers are left unchanged);
    * with ``slice count > 1``: each existing trigger's field is subdivided into
      ``N`` evenly-spaced slice triggers, ``metadata.triggers`` is replaced,
      ``artifact_length`` becomes the per-slice length and the offset becomes 0
      (each new trigger sits at its slice start).

    Parameters
    ----------
    channel : str | int | None
        Channel to display. Name (str), index (int), or ``None`` for the first
        EEG channel.
    n_epochs : int
        Number of artifact epochs to overlay (default: 5).
    initial_offset : float | None
        Starting offset in seconds. ``None`` uses the current context value.
    artifact_length : int | None
        Starting artifact length in samples. ``None`` uses the current context
        value.
    slice_count : int
        Initial number of slice triggers per field (1 = no subdivision).
    settings_path : str
        JSON file used by the Save/Load buttons. Defaults to
        ``"output/trigger_editor_settings.json"`` — the gitignored ``output/``
        directory — and the parent folder is created on save if needed.
    """

    name = "trigger_editor"
    description = "Interactively edit offset, artifact length and slice triggers"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        channel: str | int | None = None,
        n_epochs: int = 5,
        initial_offset: float | None = None,
        artifact_length: int | None = None,
        slice_count: int = 1,
        settings_path: str = "output/trigger_editor_settings.json",
    ) -> None:
        self.channel = channel
        self.n_epochs = n_epochs
        self.initial_offset = initial_offset
        self.artifact_length = artifact_length
        self.slice_count = int(slice_count)
        self.settings_path = settings_path
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None and self.artifact_length is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector first or pass artifact_length."
            )
        if len(context.get_triggers()) < 1:
            raise ProcessorValidationError("TriggerEditor needs at least one trigger.")
        if self.slice_count < 1:
            raise ProcessorValidationError(f"slice_count must be >= 1, got {self.slice_count}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = np.asarray(context.get_triggers(), dtype=int)
        sfreq = context.get_sfreq()
        n_samples = raw.n_times
        base_length = self.artifact_length if self.artifact_length is not None else context.get_artifact_length()
        start_offset = self.initial_offset if self.initial_offset is not None else context.metadata.artifact_to_trigger_offset

        # --- COMPUTE ---
        ch_idx = self._resolve_channel(raw)
        base_duration = base_length / sfreq
        # Extract a window wide enough to edit + zoom out around the field.
        padding = base_duration * 0.5
        pre_span = base_duration * 1.0 + padding
        post_span = base_duration * 3.0 + padding
        epochs_data, time_axis = self._extract_epochs(raw._data[ch_idx], triggers, sfreq, pre_span, post_span)

        result = self._run_editor(
            epochs_data,
            time_axis,
            raw.ch_names[ch_idx],
            sfreq,
            start_offset=float(start_offset),
            start_length=float(base_duration),
            start_count=self.slice_count,
        )
        if result is None:
            logger.info("Trigger editor cancelled; context unchanged")
            return context

        offset_s = float(result["offset"])
        length_s = float(result["length"])
        count = max(1, int(result["count"]))

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        if count > 1:
            line_pos = self._slice_line_positions(offset_s, length_s, count)  # seconds rel. to trigger
            line_samples = np.round(np.asarray(line_pos) * sfreq).astype(int)
            new_triggers = np.concatenate([triggers + int(s) for s in line_samples])
            new_triggers = np.unique(np.clip(new_triggers, 0, n_samples - 1))
            new_metadata.triggers = np.sort(new_triggers)
            new_metadata.artifact_length = max(1, int(round((length_s / count) * sfreq)))
            new_metadata.artifact_to_trigger_offset = 0.0
            new_metadata.slices_per_volume = count
            logger.info(
                "Trigger editor: {} slice triggers ({} per field), artifact_length={} samples, offset=0",
                len(new_metadata.triggers),
                count,
                new_metadata.artifact_length,
            )
        else:
            new_metadata.artifact_to_trigger_offset = offset_s
            new_metadata.artifact_length = max(1, int(round(length_s * sfreq)))
            logger.info(
                "Trigger editor: offset={:.6f} s, artifact_length={} samples",
                offset_s,
                new_metadata.artifact_length,
            )

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    # ----------------------------------------------------------------- #
    # Pure helpers (unit-testable, no GUI)                               #
    # ----------------------------------------------------------------- #

    @staticmethod
    def _slice_line_positions(offset: float, length: float, count: int) -> np.ndarray:
        """Evenly-spaced slice-trigger positions (s) inside the artifact field.

        ``count`` lines tile ``[offset, offset + length]`` start-aligned, i.e.
        line ``k`` sits at ``offset + k * length / count`` (each slice's start).
        """
        count = max(1, int(count))
        return offset + np.arange(count) * (length / count)

    @staticmethod
    def _zoom_xlim(lo: float, hi: float, cursor: float, factor: float) -> tuple[float, float]:
        """New x-limits for a scroll-zoom that keeps ``cursor`` fixed.

        ``factor < 1`` zooms in, ``factor > 1`` zooms out.
        """
        return cursor - (cursor - lo) * factor, cursor + (hi - cursor) * factor

    @staticmethod
    def _hit_test(x: float, offset: float, length: float, edge_tol: float) -> str | None:
        """Classify a click at ``x`` against the artifact window.

        Returns ``"left"``/``"right"`` when within ``edge_tol`` of an edge,
        ``"body"`` when strictly inside, else ``None``.
        """
        start, end = offset, offset + length
        if abs(x - start) <= edge_tol:
            return "left"
        if abs(x - end) <= edge_tol:
            return "right"
        if start < x < end:
            return "body"
        return None

    @staticmethod
    def _settings_to_dict(offset: float, length: float, count: int) -> dict:
        """Serialise editor settings."""
        return {"offset_s": float(offset), "length_s": float(length), "slice_count": int(count)}

    @staticmethod
    def _settings_from_dict(data: dict) -> tuple[float, float, int]:
        """Deserialise editor settings -> ``(offset_s, length_s, slice_count)``."""
        return float(data["offset_s"]), float(data["length_s"]), max(1, int(data.get("slice_count", 1)))

    def _save_settings(self, path: str, offset: float, length: float, count: int) -> None:
        """Write the current settings to ``path`` as JSON (creating the dir)."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._settings_to_dict(offset, length, count), fh, indent=2)

    def _load_settings(self, path: str) -> tuple[float, float, int] | None:
        """Read settings from ``path``; return ``None`` if the file is absent."""
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return self._settings_from_dict(json.load(fh))

    def _resolve_channel(self, raw: mne.io.Raw) -> int:
        """Determine the channel index to display."""
        if self.channel is None:
            eeg_channels = mne.pick_types(raw.info, eeg=True, exclude="bads")
            if len(eeg_channels) == 0:
                raise ProcessorError("No EEG channels found in raw data.")
            return int(eeg_channels[0])
        if isinstance(self.channel, int):
            return self.channel
        return raw.ch_names.index(self.channel)

    def _extract_epochs(
        self,
        channel_data: np.ndarray,
        triggers: np.ndarray,
        sfreq: float,
        pre_span: float,
        post_span: float,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Extract mean-subtracted epochs centred on the first few triggers."""
        n_pre = int(pre_span * sfreq)
        n_post = int(post_span * sfreq)
        n_total = n_pre + n_post
        time_axis = (np.arange(n_total) - n_pre) / sfreq

        n_show = min(self.n_epochs, len(triggers))
        epochs: list[np.ndarray] = []
        for i in range(n_show):
            start = int(triggers[i]) - n_pre
            end = start + n_total
            if start < 0 or end > len(channel_data):
                continue
            epoch = channel_data[start:end].astype(float)
            epoch -= np.mean(epoch)
            epochs.append(epoch)

        if not epochs:
            raise ProcessorError("Could not extract any epochs for display.")
        return epochs, time_axis

    # ----------------------------------------------------------------- #
    # Interactive GUI                                                    #
    # ----------------------------------------------------------------- #

    def _run_editor(
        self,
        epochs_data: list[np.ndarray],
        time_axis: np.ndarray,
        ch_name: str,
        sfreq: float,
        start_offset: float,
        start_length: float,
        start_count: int,
    ) -> dict | None:
        """Show the editor; return ``{offset, length, count}`` or ``None`` if cancelled."""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.subplots_adjust(bottom=0.30, top=0.94)
        for epoch in epochs_data:
            ax.plot(time_axis, epoch, alpha=0.5, linewidth=0.7)
        ax.axvline(0, color="blue", ls="--", lw=1.5, label="Trigger")
        ax.set_xlabel("Time relative to trigger (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Trigger Editor — {ch_name}")
        ax.grid(True, alpha=0.3)

        t_lo, t_hi = float(time_axis[0]), float(time_axis[-1])
        state: dict = {
            "offset": float(start_offset),
            "length": float(start_length),
            "count": max(1, int(start_count)),
            "confirmed": False,
            "drag": None,
            "press_x": 0.0,
            "press_offset": 0.0,
            "press_length": 0.0,
            "shade": None,
            "edge_l": None,
            "edge_r": None,
            "slice_lines": [],
            "syncing": False,
        }
        info_text = ax.text(
            0.02,
            0.97,
            "",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
        )

        def _clear_overlay() -> None:
            for key in ("shade", "edge_l", "edge_r"):
                if state[key] is not None:
                    state[key].remove()
                    state[key] = None
            for line in state["slice_lines"]:
                line.remove()
            state["slice_lines"] = []

        def _draw_overlay() -> None:
            _clear_overlay()
            o, length, count = state["offset"], state["length"], state["count"]
            state["shade"] = ax.axvspan(o, o + length, alpha=0.12, color="red", label="_nolegend_")
            state["edge_l"] = ax.axvline(o, color="red", lw=2)
            state["edge_r"] = ax.axvline(o + length, color="darkred", lw=2, ls=":")
            if count > 1:
                for pos in self._slice_line_positions(o, length, count):
                    state["slice_lines"].append(ax.axvline(float(pos), color="green", lw=1.0, alpha=0.7))
            info_text.set_text(
                f"Offset: {o * 1000:.2f} ms\nLength: {length * 1000:.2f} ms\nSlices: {count}"
            )

        def _sync_textboxes() -> None:
            state["syncing"] = True
            try:
                tb_offset.set_val(f"{state['offset'] * 1000:.2f}")
                tb_length.set_val(f"{state['length'] * 1000:.2f}")
                tb_slices.set_val(str(state["count"]))
            finally:
                state["syncing"] = False

        def _refresh(sync: bool = True) -> None:
            _draw_overlay()
            if sync:
                _sync_textboxes()
            fig.canvas.draw_idle()

        # Initial framing around the artifact window.
        margin = max(0.3 * state["length"], 1.0 / sfreq)
        ax.set_xlim(
            max(min(state["offset"], 0.0) - margin, t_lo),
            min(max(state["offset"] + state["length"], 0.0) + margin, t_hi),
        )

        # --- Text fields ---
        tb_offset = TextBox(fig.add_axes([0.12, 0.17, 0.08, 0.045]), "Offset(ms) ", initial=f"{state['offset'] * 1000:.2f}")
        tb_length = TextBox(fig.add_axes([0.34, 0.17, 0.08, 0.045]), "Len(ms) ", initial=f"{state['length'] * 1000:.2f}")
        tb_slices = TextBox(fig.add_axes([0.56, 0.17, 0.06, 0.045]), "Slices ", initial=str(state["count"]))

        def _on_offset_submit(text: str) -> None:
            if state["syncing"]:
                return
            try:
                state["offset"] = float(text) / 1000.0
            except ValueError:
                return
            _refresh(sync=False)

        def _on_length_submit(text: str) -> None:
            if state["syncing"]:
                return
            try:
                state["length"] = max(1.0 / sfreq, float(text) / 1000.0)
            except ValueError:
                return
            _refresh(sync=False)

        def _on_slices_submit(text: str) -> None:
            if state["syncing"]:
                return
            try:
                state["count"] = max(1, int(float(text)))
            except ValueError:
                return
            _refresh(sync=False)

        tb_offset.on_submit(_on_offset_submit)
        tb_length.on_submit(_on_length_submit)
        tb_slices.on_submit(_on_slices_submit)

        # --- Drag the window (body = move offset, edges = change length) ---
        def _on_press(event) -> None:
            if event.inaxes != ax or event.button != 1 or event.xdata is None:
                return
            xlo, xhi = ax.get_xlim()
            tol = 0.02 * (xhi - xlo)
            region = self._hit_test(event.xdata, state["offset"], state["length"], tol)
            if region is not None:
                state["drag"] = region
                state["press_x"] = event.xdata
                state["press_offset"] = state["offset"]
                state["press_length"] = state["length"]

        def _on_motion(event) -> None:
            if state["drag"] is None or event.inaxes != ax or event.xdata is None:
                return
            dx = event.xdata - state["press_x"]
            min_len = 1.0 / sfreq
            if state["drag"] == "body":
                state["offset"] = state["press_offset"] + dx
            elif state["drag"] == "left":
                end = state["press_offset"] + state["press_length"]
                state["offset"] = min(state["press_offset"] + dx, end - min_len)
                state["length"] = end - state["offset"]
            elif state["drag"] == "right":
                state["length"] = max(min_len, state["press_length"] + dx)
            _refresh(sync=True)

        def _on_release(_event) -> None:
            state["drag"] = None

        fig.canvas.mpl_connect("button_press_event", _on_press)
        fig.canvas.mpl_connect("motion_notify_event", _on_motion)
        fig.canvas.mpl_connect("button_release_event", _on_release)

        # --- Scroll-wheel zoom around the cursor ---
        def _on_scroll(event) -> None:
            if event.inaxes != ax or event.xdata is None:
                return
            factor = 0.8 if event.button == "up" else 1.25
            xlo, xhi = ax.get_xlim()
            new_lo, new_hi = self._zoom_xlim(xlo, xhi, event.xdata, factor)
            ax.set_xlim(new_lo, new_hi)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("scroll_event", _on_scroll)

        # --- Buttons: Save / Load / Confirm / Cancel ---
        def _close_fig() -> None:
            with contextlib.suppress(Exception):
                fig.canvas.manager.destroy()
            plt.close(fig)

        def _on_save(_event) -> None:
            try:
                self._save_settings(self.settings_path, state["offset"], state["length"], state["count"])
                logger.info("Saved trigger-editor settings to {}", self.settings_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not save settings: {}", exc)

        def _on_load(_event) -> None:
            try:
                loaded = self._load_settings(self.settings_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not load settings: {}", exc)
                return
            if loaded is None:
                logger.warning("No settings file found at {}", self.settings_path)
                return
            state["offset"], state["length"], state["count"] = loaded
            _refresh(sync=True)

        def _on_confirm(_event) -> None:
            state["confirmed"] = True
            _close_fig()

        save_btn = Button(fig.add_axes([0.12, 0.04, 0.08, 0.05]), "Save")
        load_btn = Button(fig.add_axes([0.21, 0.04, 0.08, 0.05]), "Load")
        confirm_btn = Button(fig.add_axes([0.72, 0.04, 0.08, 0.05]), "Confirm")
        cancel_btn = Button(fig.add_axes([0.81, 0.04, 0.08, 0.05]), "Cancel")
        save_btn.on_clicked(_on_save)
        load_btn.on_clicked(_on_load)
        confirm_btn.on_clicked(_on_confirm)
        cancel_btn.on_clicked(lambda _e: _close_fig())

        _refresh(sync=False)

        console = get_console()
        console.set_active_prompt("Edit: drag window/edges, scroll to zoom, fields for exact values, then Confirm")
        try:
            with suspend_raw_mode():
                show_matplotlib_figure(fig)
        finally:
            plt.close(fig)
            console.clear_active_prompt()

        if not state["confirmed"]:
            return None
        return {"offset": state["offset"], "length": state["length"], "count": state["count"]}
