"""
Visualization Processors Module

This module contains processors for generating visual diagnostics of pipeline
results, including Matplotlib and MNE-Python based visualisations.

Author: FACETpy Team
Date: 2025-01-12
"""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mne
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from scipy import signal

from ..console import suspend_raw_mode
from ..core import ProcessingContext, Processor, register_processor
from ..helpers.plotting import show_matplotlib_figure

# Valid signal sources for RawPlotter
SOURCE_RAW = "raw"
SOURCE_PREDICTION = "prediction"
_VALID_SOURCES = (SOURCE_RAW, SOURCE_PREDICTION)


@register_processor
class RawPlotter(Processor):
    """Plot raw EEG data snippets during the pipeline.

    Supports both Matplotlib-based summary figures as well as native MNE-Python
    interactive plots. By default, a Matplotlib plot is generated and saved to
    the configured path, overlaying the current corrected signal with the
    original recording for quick visual inspection.

    Parameters
    ----------
    mode : str, optional
        Plotting backend: ``'matplotlib'`` (default) or ``'mne'``.
    channel : str or int, optional
        Single channel to visualise (name or index). Defaults to the first EEG
        channel, falling back to the first channel overall.
    start : float, optional
        Start time in seconds of the snippet to plot (default: 0.0).
    duration : float, optional
        Duration in seconds of the snippet to plot (default: 10.0).
    overlay_original : bool, optional
        Overlay original recording when available (default: True). Semantics
        depend on ``source``:

        * ``source='raw'``        — overlays the **original recording** on top
          of the current (corrected) signal: the classical before/after view.
        * ``source='prediction'`` — overlays the **original noisy recording**
          on top of the predicted artifact: useful for residual diagnostics —
          where the two curves diverge is where the model's prediction misses
          part of the artifact (= residual artifact in the corrected signal).
    scale : float, optional
        Multiplier applied to amplitude values (default: 1e6 for V → µV).
    save_path : str or Path, optional
        File path to save the generated plot.
    show : bool, optional
        Whether to display the plot interactively (default: False).
    auto_close : bool, optional
        Close the figure after saving when running headless (default: True).
    figure_kwargs : dict, optional
        Additional keyword arguments forwarded to ``plt.subplots()``.
    mne_kwargs : dict, optional
        Additional keyword arguments forwarded to ``mne.io.Raw.plot()``.
    picks : sequence of int or str, optional
        Explicit channel picks for MNE plotting mode.
    title : str, optional
        Custom figure title for Matplotlib mode.
    source : str, optional
        Which signal to plot (default: ``'raw'``):

        * ``'raw'``        — current ``context.get_raw()`` data (corrected signal).
        * ``'prediction'`` — ``context.get_estimated_noise()`` (the predicted
          artifact written by a ``DeepLearningCorrection`` or compatible
          processor). Skipped with a warning if no prediction is present.
    """

    name = "raw_plotter"
    description = "Plot raw data snippets for visual inspection."
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        mode: str = "matplotlib",
        channel: str | int | None = None,
        start: float = 0.0,
        duration: float = 10.0,
        overlay_original: bool = True,
        scale: float = 1e6,
        save_path: str | Path | None = None,
        show: bool = False,
        auto_close: bool = True,
        figure_kwargs: dict[str, Any] | None = None,
        mne_kwargs: dict[str, Any] | None = None,
        picks: Sequence[int | str] | None = None,
        title: str | None = None,
        source: str = SOURCE_RAW,
    ) -> None:
        self.mode = mode.lower()
        self.channel = channel
        self.start = max(0.0, start)
        self.duration = duration
        self.overlay_original = overlay_original
        self.scale = scale
        self.save_path = Path(save_path) if save_path else None
        self.show = show
        self.auto_close = auto_close
        self.figure_kwargs = figure_kwargs or {}
        self.mne_kwargs = mne_kwargs or {}
        self.picks = picks
        self.title = title
        source_lower = source.lower()
        if source_lower not in _VALID_SOURCES:
            raise ValueError(f"Unsupported source '{source}'. Valid options: {list(_VALID_SOURCES)}")
        self.source = source_lower
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        if raw is None:
            logger.warning("No raw data available; skipping plot generation.")
            return context

        # --- RESOLVE SOURCE ---
        data = self._resolve_source_data(context, raw)
        if data is None:
            return context

        # --- LOG ---
        logger.info("Generating {} plot for source='{}'", self.mode, self.source)

        # --- COMPUTE ---
        if self.mode == "mne":
            self._plot_with_mne(raw, data)
        elif self.mode == "matplotlib":
            self._plot_with_matplotlib(raw, data, context)
        else:
            logger.error("Unknown plotting mode '{}'. Skipping plot.", self.mode)

        # --- RETURN ---
        return context

    def _resolve_source_data(self, context: ProcessingContext, raw) -> np.ndarray | None:
        """Return the (n_channels, n_times) array to plot, or ``None`` to skip.

        Centralises the ``source`` parameter so MNE and Matplotlib modes share
        the same selection logic and skip rules.
        """
        if self.source == SOURCE_RAW:
            return raw._data
        if self.source == SOURCE_PREDICTION:
            if not context.has_estimated_noise():
                logger.warning(
                    "RawPlotter source='prediction' requested but no estimated noise "
                    "is present in the context. Place this step AFTER a "
                    "DeepLearningCorrection (or any processor that populates "
                    "context.get_estimated_noise()). Skipping plot."
                )
                return None
            data = context.get_estimated_noise()
            if data.shape != raw._data.shape:
                logger.warning(
                    "Predicted artifact shape {} does not match raw {}; skipping plot.",
                    data.shape,
                    raw._data.shape,
                )
                return None
            return data
        logger.error("Unknown source '{}'; skipping plot.", self.source)
        return None

    def _plot_with_mne(self, raw, data: np.ndarray) -> None:
        """Use mne.io.Raw.plot to visualise the resolved source array."""
        # For non-raw sources, wrap the data in a temporary RawArray that
        # shares the channel metadata of the context's raw object.
        if self.source != SOURCE_RAW:
            raw = mne.io.RawArray(data, raw.info.copy(), verbose="WARNING")

        plot_kwargs: dict[str, Any] = dict(
            start=self.start,
            duration=self.duration,
            show=self.show,
        )
        if self.picks is not None:
            plot_kwargs["picks"] = self.picks

        plot_kwargs.update(self.mne_kwargs)

        logger.info(
            "Generating MNE-Python plot (start=%.2fs, duration=%.2fs, picks=%s)",
            self.start,
            self.duration,
            self.picks,
        )

        plot_kwargs["block"] = False
        fig = raw.plot(**plot_kwargs)

        # MNE returns different types depending on backend:
        # - matplotlib backend: Figure with savefig()
        # - Qt backend (mne-qt-browser): MNEQtBrowser without savefig()
        is_matplotlib_figure = hasattr(fig, "savefig")

        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            if is_matplotlib_figure:
                fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
                logger.info("Saved MNE plot to {}", self.save_path)
            else:
                logger.warning(
                    "MNE returned {} (Qt backend); savefig not supported. "
                    "Use mode='matplotlib' or mne.viz.set_browser_backend('matplotlib') for saving.",
                    type(fig).__name__,
                )
                return

        if self.show:
            if is_matplotlib_figure:
                with suspend_raw_mode():
                    plt.show(block=False)
                    while plt.fignum_exists(fig.number):
                        fig.canvas.flush_events()
                        time.sleep(0.05)
            # Qt backend handles display independently; nothing to do
        elif (self.auto_close or self.save_path) and is_matplotlib_figure:
            plt.close(fig)

    def _plot_with_matplotlib(self, raw, data: np.ndarray, context: ProcessingContext) -> None:
        """Use Matplotlib to plot the resolved source array, optionally with overlay."""
        channel_idx, channel_name = self._resolve_channel(raw)
        sfreq = raw.info["sfreq"]
        n_times = data.shape[1]
        start_sample = int(self.start * sfreq)
        stop_sample = start_sample + int(self.duration * sfreq) if self.duration > 0 else n_times
        stop_sample = min(stop_sample, n_times)

        if stop_sample <= start_sample:
            stop_sample = n_times

        times = np.arange(start_sample, stop_sample) / sfreq
        current = data[channel_idx, start_sample:stop_sample]

        # Overlay semantics:
        # - source="raw":        overlay = original recording  (before/after view)
        # - source="prediction": overlay = original NOISY signal (residual diagnostic)
        original = self._extract_original_overlay(context, channel_idx, times, sfreq) if self.overlay_original else None

        fig_kwargs = {"figsize": (12, 4)}
        fig_kwargs.update(self.figure_kwargs)
        fig, ax = plt.subplots(**fig_kwargs)

        if self.source == SOURCE_PREDICTION:
            current_label = "Predicted artifact"
            current_color = "#dc2626"  # red — model output
            current_alpha = 0.9
            original_label = "Original noisy"
            original_color = "#374151"  # dark gray — reference
            original_alpha = 0.55
        else:
            current_label = "Corrected"
            current_color = None  # let matplotlib choose
            current_alpha = 0.8
            original_label = "Original"
            original_color = None
            original_alpha = 0.6

        # Plot original FIRST so the foreground line is on top.
        if original is not None and self.source == SOURCE_PREDICTION:
            ax.plot(
                times,
                original * self.scale,
                label=original_label,
                color=original_color,
                alpha=original_alpha,
                linewidth=1.0,
            )
        ax.plot(
            times,
            current * self.scale,
            label=current_label,
            color=current_color,
            alpha=current_alpha,
            linewidth=1.0,
        )
        # For source="raw" the original goes on top (preserves prior ordering).
        if original is not None and self.source != SOURCE_PREDICTION:
            ax.plot(
                times,
                original * self.scale,
                label=original_label,
                color=original_color,
                alpha=original_alpha,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (uV)")
        ax.grid(True, alpha=0.2)
        if original is not None:
            ax.legend(loc="upper right")

        default_title = (
            f"{channel_name} – predicted artifact vs original ({self.duration:.1f}s)"
            if self.source == SOURCE_PREDICTION and original is not None
            else f"{channel_name} – predicted artifact ({self.duration:.1f}s snippet)"
            if self.source == SOURCE_PREDICTION
            else f"{channel_name} – {self.duration:.1f}s snippet"
        )
        ax.set_title(self.title or default_title)

        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved Matplotlib plot to {}", self.save_path)

        if self.show:
            with suspend_raw_mode():
                show_matplotlib_figure(fig)
            plt.close(fig)
        elif self.auto_close or self.save_path:
            plt.close(fig)

    def _extract_original_overlay(
        self,
        context: ProcessingContext,
        channel_idx: int,
        times: np.ndarray,
        sfreq: float,
    ) -> np.ndarray | None:
        """Retrieve original-recording data for overlay, resampling if needed.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context.
        channel_idx : int
            Index of the channel to extract.
        times : np.ndarray
            Time axis of the corrected snippet (used to determine length).
        sfreq : float
            Sampling frequency of the current (corrected) recording.

        Returns
        -------
        np.ndarray or None
            Original data array aligned to ``times``, or ``None`` when unavailable.
        """
        if not self.overlay_original:
            return None

        raw_original = context.get_raw_original()
        if raw_original is None:
            logger.warning("Original data unavailable; skipping overlay.")
            return None

        sfreq_original = raw_original.info["sfreq"]
        start_original = int(self.start * sfreq_original)
        stop_original = (
            start_original + int(self.duration * sfreq_original) if self.duration > 0 else raw_original.n_times
        )
        stop_original = min(stop_original, raw_original.n_times)

        if stop_original <= start_original:
            stop_original = raw_original.n_times

        if channel_idx >= len(raw_original.ch_names):
            logger.warning("Channel index out of range for original data; skipping overlay.")
            return None

        try:
            original_data = raw_original.get_data(
                picks=[channel_idx],
                start=start_original,
                stop=stop_original,
            )
        except (IndexError, ValueError) as exc:
            logger.warning("Failed to extract original data: {}; skipping overlay.", exc)
            return None

        if original_data.size == 0:
            logger.warning("Original data returned empty array; skipping overlay.")
            return None

        original = original_data[0]

        if sfreq_original != sfreq:
            if len(original) <= 1:
                logger.warning(
                    "Original data too short for overlay (length {} vs {}); skipping overlay.",
                    len(original),
                    len(times),
                )
                return None
            original = signal.resample(original, len(times))
        elif len(original) != len(times):
            logger.warning(
                "Original data length mismatch (length {} vs {}); skipping overlay.",
                len(original),
                len(times),
            )
            return None

        return original

    def _resolve_channel(self, raw) -> tuple:
        """Resolve channel selection to index and name.

        Parameters
        ----------
        raw : mne.io.Raw
            The Raw object to look up channel information from.

        Returns
        -------
        tuple of (int, str)
            Channel index and channel name.
        """
        if self.channel is None:
            for idx, ch_type in enumerate(raw.get_channel_types()):
                if ch_type == "eeg":
                    return idx, raw.ch_names[idx]
            return 0, raw.ch_names[0]

        if isinstance(self.channel, int):
            idx = max(0, min(self.channel, len(raw.ch_names) - 1))
            return idx, raw.ch_names[idx]

        if isinstance(self.channel, str):
            try:
                idx = raw.ch_names.index(self.channel)
                return idx, raw.ch_names[idx]
            except ValueError:
                logger.warning(
                    "Requested channel '{}' not found. Falling back to first channel.",
                    self.channel,
                )
                return 0, raw.ch_names[0]

        logger.warning("Unsupported channel specifier {}. Using first channel.", self.channel)
        return 0, raw.ch_names[0]
