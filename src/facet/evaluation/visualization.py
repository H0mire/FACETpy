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

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from scipy import signal

from ..console import suspend_raw_mode
from ..core import ProcessingContext, Processor, register_processor


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
        Overlay original recording when available (default: True).
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
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        if raw is None:
            logger.warning("No raw data available; skipping plot generation.")
            return context

        # --- LOG ---
        logger.info("Generating {} plot", self.mode)

        # --- COMPUTE ---
        if self.mode == "mne":
            self._plot_with_mne(raw)
        elif self.mode == "matplotlib":
            self._plot_with_matplotlib(raw, context)
        else:
            logger.error("Unknown plotting mode '{}'. Skipping plot.", self.mode)

        # --- RETURN ---
        return context

    def _plot_with_mne(self, raw) -> None:
        """Use mne.io.Raw.plot to visualise the data."""
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

        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved MNE plot to {}", self.save_path)

        if self.show:
            with suspend_raw_mode():
                plt.show(block=False)
                while plt.fignum_exists(fig.number):
                    fig.canvas.flush_events()
                    time.sleep(0.05)
        elif self.auto_close or self.save_path:
            plt.close(fig)

    def _plot_with_matplotlib(self, raw, context: ProcessingContext) -> None:
        """Use Matplotlib to create a before/after comparison plot."""
        channel_idx, channel_name = self._resolve_channel(raw)
        sfreq = raw.info["sfreq"]
        start_sample = int(self.start * sfreq)
        stop_sample = start_sample + int(self.duration * sfreq) if self.duration > 0 else raw.n_times
        stop_sample = min(stop_sample, raw.n_times)

        if stop_sample <= start_sample:
            stop_sample = raw.n_times

        times = np.arange(start_sample, stop_sample) / sfreq
        current = raw.get_data(picks=[channel_idx], start=start_sample, stop=stop_sample)[0]

        original = self._extract_original_overlay(context, channel_idx, times, sfreq)

        fig_kwargs = {"figsize": (12, 4)}
        fig_kwargs.update(self.figure_kwargs)
        fig, ax = plt.subplots(**fig_kwargs)

        ax.plot(times, current * self.scale, label="Corrected", alpha=0.8)
        if original is not None:
            ax.plot(times, original * self.scale, label="Original", alpha=0.6)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (uV)")
        ax.grid(True, alpha=0.2)
        if original is not None:
            ax.legend(loc="upper right")

        title = self.title or f"{channel_name} – {self.duration:.1f}s snippet"
        ax.set_title(title)

        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved Matplotlib plot to {}", self.save_path)

        if self.show:
            with suspend_raw_mode():
                plt.show(block=False)
                while plt.fignum_exists(fig.number):
                    fig.canvas.flush_events()
                    time.sleep(0.05)
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
