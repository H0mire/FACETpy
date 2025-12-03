"""
Visualization Processors Module

This module contains processors for generating visual diagnostics of pipeline
results, including Matplotlib and MNE-Python based visualisations.

Author: FACETpy Team
Date: 2025-01-12
"""

from pathlib import Path
from typing import Optional, Union, Sequence, Dict, Any

import numpy as np
from loguru import logger

from ..core import Processor, ProcessingContext, register_processor


@register_processor
class RawPlotter(Processor):
    """
    Plot raw EEG data snippets during the pipeline.

    Supports both Matplotlib-based summary figures as well as native MNE-Python
    interactive plots. By default, a Matplotlib plot is generated and saved to
    the configured path, overlaying the current corrected signal with the
    original recording for quick visual inspection.

    Example:
        plotter = RawPlotter(
            mode=\"matplotlib\",
            channel=\"Fp1\",
            duration=15,
            overlay_original=True,
            save_path=\"output/pipeline_plot.png\"
        )
    """

    name = "raw_plotter"
    description = "Plot raw data snippets for visual inspection."
    modifies_raw = False

    def __init__(
        self,
        mode: str = "matplotlib",
        channel: Optional[Union[str, int]] = None,
        start: float = 0.0,
        duration: float = 10.0,
        overlay_original: bool = True,
        scale: float = 1e6,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        auto_close: bool = True,
        figure_kwargs: Optional[Dict[str, Any]] = None,
        mne_kwargs: Optional[Dict[str, Any]] = None,
        picks: Optional[Sequence[Union[int, str]]] = None,
        title: Optional[str] = None
    ):
        """
        Initialize the plotter.

        Args:
            mode: Plotting mode ('matplotlib' or 'mne').
            channel: Single channel to visualise (name or index). Default is the first EEG.
            start: Start time (seconds) of the snippet to plot.
            duration: Duration (seconds) of the snippet to plot.
            overlay_original: Overlay original recording when available.
            scale: Multiplier for converting values (default 1e6 for volts->µV).
            save_path: Optional file path to save the plot.
            show: Whether to display the plot interactively.
            auto_close: Close the figure after saving when running headless.
            figure_kwargs: Additional Matplotlib figure kwargs.
            mne_kwargs: Additional kwargs forwarded to ``mne.io.Raw.plot``.
            picks: Explicit channel picks for MNE plotting mode.
            title: Custom title for Matplotlib mode.
        """
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
        """Generate the configured plot."""
        raw = context.get_raw()
        if raw is None:
            logger.warning("No raw data available; skipping plot generation.")
            return context

        if self.mode == "mne":
            self._plot_with_mne(raw)
        elif self.mode == "matplotlib":
            self._plot_with_matplotlib(raw, context)
        else:
            logger.error(f"Unknown plotting mode '{self.mode}'. Skipping plot.")

        return context

    def _plot_with_mne(self, raw):
        """Use mne.io.Raw.plot to visualise the data."""
        plot_kwargs = dict(
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
            self.picks
        )

        fig = raw.plot(block=not self.auto_close,**plot_kwargs)

        if self.save_path:
            from matplotlib import pyplot as plt  # Lazy import
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved MNE plot to {self.save_path}")
            if self.auto_close:
                plt.close(fig)

        if not self.show and self.auto_close and not self.save_path:
            # Close figures to avoid accumulating windows in headless runs
            from matplotlib import pyplot as plt
            plt.close(fig)

    def _plot_with_matplotlib(self, raw, context: ProcessingContext):
        """Use Matplotlib to create a before/after comparison plot."""
        from matplotlib import pyplot as plt  # Lazy import

        channel_idx, channel_name = self._resolve_channel(raw)
        sfreq = raw.info["sfreq"]
        start_sample = int(self.start * sfreq)
        stop_sample = start_sample + int(self.duration * sfreq) if self.duration > 0 else raw.n_times
        stop_sample = min(stop_sample, raw.n_times)

        if stop_sample <= start_sample:
            stop_sample = raw.n_times

        times = np.arange(start_sample, stop_sample) / sfreq
        current = raw.get_data(picks=[channel_idx], start=start_sample, stop=stop_sample)[0]

        original = None
        if self.overlay_original:
            raw_original = context.get_raw_original()
            if raw_original is not None:
                # Calculate indices for original data based on its own sampling rate
                sfreq_original = raw_original.info["sfreq"]
                start_sample_original = int(self.start * sfreq_original)
                stop_sample_original = start_sample_original + int(self.duration * sfreq_original) if self.duration > 0 else raw_original.n_times
                stop_sample_original = min(stop_sample_original, raw_original.n_times)
                
                # Ensure valid range
                if stop_sample_original <= start_sample_original:
                    stop_sample_original = raw_original.n_times
                
                # Check if channel exists in original data
                if channel_idx < len(raw_original.ch_names):
                    try:
                        original_data = raw_original.get_data(
                            picks=[channel_idx],
                            start=start_sample_original,
                            stop=stop_sample_original
                        )
                        if original_data.size > 0:
                            original = original_data[0]
                            # Resample original to match current times if sampling rates differ
                            if sfreq_original != sfreq and len(original) > 0:
                                from scipy import signal
                                if len(original) > 1:
                                    original = signal.resample(original, len(times))
                                else:
                                    # If original is too short, skip overlay
                                    logger.warning(
                                        "Original data too short for overlay (length %d vs %d); skipping overlay.",
                                        len(original), len(times)
                                    )
                                    original = None
                            # Also check if lengths match even with same sampling rate
                            elif len(original) != len(times):
                                logger.warning(
                                    "Original data length mismatch (length %d vs %d); skipping overlay.",
                                    len(original), len(times)
                                )
                                original = None
                        else:
                            logger.warning("Original data returned empty array; skipping overlay.")
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Failed to extract original data: {e}; skipping overlay.")
                else:
                    logger.warning("Channel index out of range for original data; skipping overlay.")
            else:
                logger.warning("Original data unavailable; skipping overlay.")

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
            logger.info(f"Saved Matplotlib plot to {self.save_path}")

        if self.show:
            plt.show(block=True)
        elif self.auto_close or self.save_path:
            plt.close(fig)

    def _resolve_channel(self, raw):
        """Resolve channel selection to index and name."""
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
                    "Requested channel '%s' not found. Falling back to first channel.",
                    self.channel
                )
                return 0, raw.ch_names[0]

        logger.warning("Unsupported channel specifier %s. Using first channel.", self.channel)
        return 0, raw.ch_names[0]
