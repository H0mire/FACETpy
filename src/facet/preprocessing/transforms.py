"""
Simple raw-transform processors.

Contains small, focused processors for common in-pipeline data manipulations
that don't fit neatly into filtering, resampling, or trigger handling.
"""

import contextlib
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import mne
from loguru import logger
from matplotlib.widgets import Button, SpanSelector

from ..console import get_console, suspend_raw_mode
from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor
from ..helpers.plotting import show_matplotlib_figure


@register_processor
class Crop(Processor):
    """Crop the Raw recording to a time interval.

    A concise alternative to ``LambdaProcessor`` for the common pattern of
    restricting the recording to a specific window before processing.

    Parameters
    ----------
    tmin : float, optional
        Start time in seconds.  ``None`` keeps the original start.
    tmax : float, optional
        End time in seconds.  ``None`` keeps the original end.

        If both ``tmin`` and ``tmax`` are ``None``, an interactive
        Matplotlib selector is opened to choose the crop window.

    Examples
    --------
    ::

        Crop(tmin=0, tmax=162)
        Crop()  # open interactive selector
    """

    name = "crop"
    description = "Crop Raw recording to a time interval"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = False

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
    ):
        self.tmin = tmin
        self.tmax = tmax
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)

        if self.tmin is not None and self.tmax is not None and self.tmax <= self.tmin:
            raise ProcessorValidationError(f"tmax must be greater than tmin, got tmin={self.tmin}, tmax={self.tmax}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()
        resolved_tmin = self.tmin
        resolved_tmax = self.tmax

        # --- COMPUTE ---
        if self.tmin is None and self.tmax is None:
            logger.info("No crop boundaries provided; opening interactive crop selector.")
            selected_interval = self._show_interactive_crop_selector(raw)
            if selected_interval is not None:
                resolved_tmin, resolved_tmax = selected_interval
            else:
                logger.info("Interactive crop selection cancelled; keeping full recording.")

        kwargs = {}
        if resolved_tmin is not None:
            kwargs["tmin"] = resolved_tmin
        if resolved_tmax is not None:
            kwargs["tmax"] = resolved_tmax

        if resolved_tmin is not None and resolved_tmax is not None and resolved_tmax <= resolved_tmin:
            raise ProcessorValidationError(f"invalid crop interval: tmin={resolved_tmin}, tmax={resolved_tmax}")

        if kwargs:
            logger.info("Cropping raw: tmin={}, tmax={}", resolved_tmin, resolved_tmax)
            raw.crop(**kwargs)
        else:
            logger.info("Cropping skipped; no boundaries selected.")

        # --- RETURN ---
        return context.with_raw(raw)

    def _show_interactive_crop_selector(self, raw: mne.io.BaseRaw) -> tuple[float, float] | None:
        """Show interactive span selector and return selected crop bounds."""
        backend = plt.get_backend().lower()
        if "agg" in backend:
            logger.warning("Matplotlib backend '{}' is non-interactive; skipping crop selector.", backend)
            return None

        if raw.n_times < 2:
            return None

        sfreq = float(raw.info["sfreq"])
        if sfreq <= 0:
            return None

        eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        ch_idx = int(eeg_picks[0]) if len(eeg_picks) > 0 else 0
        ch_name = raw.ch_names[ch_idx]
        channel_data = raw.get_data(picks=[ch_idx])[0]
        time_axis = raw.times

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.24)

        ax.plot(time_axis, channel_data, linewidth=0.8, alpha=0.9)
        ax.set_title(f"Select crop interval - {ch_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)

        interval_state: dict[str, Any] = {
            "tmin": float(time_axis[0]),
            "tmax": float(time_axis[-1]),
            "confirmed": False,
            "shade": None,
        }

        text_label = ax.text(
            0.02,
            0.96,
            "",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        min_duration = 1.0 / sfreq

        def _refresh_overlay() -> None:
            if interval_state["shade"] is not None:
                interval_state["shade"].remove()
            interval_state["shade"] = ax.axvspan(
                interval_state["tmin"],
                interval_state["tmax"],
                facecolor="tab:blue",
                alpha=0.22,
                edgecolor="tab:blue",
                linewidth=1.0,
            )
            text_label.set_text(
                "Selected: {:.3f}s to {:.3f}s ({:.3f}s)".format(
                    interval_state["tmin"],
                    interval_state["tmax"],
                    interval_state["tmax"] - interval_state["tmin"],
                )
            )
            fig.canvas.draw_idle()

        def _on_select(xmin: float, xmax: float) -> None:
            if xmin is None or xmax is None:
                return
            left = max(0.0, min(float(xmin), float(xmax)))
            right = min(float(time_axis[-1]), max(float(xmin), float(xmax)))

            if right - left < min_duration:
                right = min(float(time_axis[-1]), left + min_duration)
                left = max(0.0, right - min_duration)

            left = round(left * sfreq) / sfreq
            right = round(right * sfreq) / sfreq
            if right <= left:
                right = min(float(time_axis[-1]), left + min_duration)
                left = max(0.0, right - min_duration)

            interval_state["tmin"] = left
            interval_state["tmax"] = right
            _refresh_overlay()

        span_selector = SpanSelector(
            ax,
            _on_select,
            "horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True,
            props=dict(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.25),
        )
        span_selector.set_active(True)

        confirm_ax = fig.add_axes([0.70, 0.06, 0.12, 0.07])
        confirm_btn = Button(confirm_ax, "Confirm")
        cancel_ax = fig.add_axes([0.84, 0.06, 0.12, 0.07])
        cancel_btn = Button(cancel_ax, "Cancel")

        def _close_fig() -> None:
            with contextlib.suppress(Exception):
                fig.canvas.manager.destroy()
            plt.close(fig)

        def _on_confirm(_) -> None:
            interval_state["confirmed"] = True
            _close_fig()

        def _on_cancel(_) -> None:
            _close_fig()

        confirm_btn.on_clicked(_on_confirm)
        cancel_btn.on_clicked(_on_cancel)
        _refresh_overlay()

        console = get_console()
        console.set_active_prompt("Drag to select crop interval, then click Confirm")
        try:
            with suspend_raw_mode():
                show_matplotlib_figure(fig)
        finally:
            plt.close(fig)
            console.clear_active_prompt()

        if not interval_state["confirmed"]:
            return None

        return float(interval_state["tmin"]), float(interval_state["tmax"])


@register_processor
class PickChannels(Processor):
    """Keep only the specified channels or channel types.

    A named, reusable alternative to the common ``lambda ctx: ctx.with_raw(
    ctx.get_raw().copy().pick(...))`` pattern.

    Parameters
    ----------
    picks : str or list of str
        Channel type (``"eeg"``, ``"stim"``, …) or list of channel
        names / types accepted by :meth:`mne.io.Raw.pick`.
    on_missing : str, optional
        Passed to MNE.  ``"ignore"`` (default) silently skips channels
        that are absent from the recording.

    Examples
    --------
    ::

        # Keep only EEG and stimulus channels
        PickChannels(picks=["eeg", "stim"])

        # Keep specific channels by name
        PickChannels(picks=["Fp1", "Fp2", "Fz"])
    """

    name = "pick_channels"
    description = "Keep only the specified channels or channel types"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(
        self,
        picks: str | list[str],
        on_missing: str = "ignore",
    ):
        self.picks = picks
        self.on_missing = on_missing
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- LOG ---
        logger.info("Picking channels: {}", self.picks)

        # --- COMPUTE + RETURN ---
        raw = context.get_raw().copy().pick(picks=self.picks, verbose=False)
        return context.with_raw(raw)


@register_processor
class DropChannels(Processor):
    """Remove named channels from the recording.

    A named, reusable alternative to the ``lambda ctx: ...drop_channels(...)``
    pattern commonly seen in inline pipeline steps.

    Parameters
    ----------
    channels : list of str
        List of channel names to remove.
    on_missing : str, optional
        ``"ignore"`` (default) skips absent names silently;
        ``"raise"`` raises an error when a channel is not found.

    Examples
    --------
    ::

        # Drop typical non-EEG channels that may be present in EDF files
        DropChannels(channels=["EKG", "EMG", "EOG", "ECG"])
    """

    name = "drop_channels"
    description = "Remove named channels from the recording"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(self, channels: list[str], on_missing: str = "ignore"):
        self.channels = channels
        self.on_missing = on_missing
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()

        # --- COMPUTE ---
        to_drop = [ch for ch in self.channels if ch in raw.ch_names] if self.on_missing == "ignore" else self.channels

        if to_drop:
            logger.info("Dropping channels: {}", to_drop)
            raw.drop_channels(to_drop)

        # --- RETURN ---
        return context.with_raw(raw)


@register_processor
class PrintMetric(Processor):
    """Print one or more evaluation metric values — useful for debugging pipelines.

    Inserts a transparent logging step that reads from the shared metrics dict
    populated by evaluation processors (e.g. :class:`~facet.evaluation.SNRCalculator`).
    The context is returned unchanged.

    Parameters
    ----------
    *metric_names : str
        One or more metric names to print (e.g. ``'snr'``, ``'rms_ratio'``).
    label : str, optional
        Optional prefix shown in brackets, e.g. ``"after PCA"``.

    Examples
    --------
    ::

        pipeline = Pipeline([
            ...,
            SNRCalculator(),
            PrintMetric("snr"),          # → "  snr=12.345"
            PCACorrection(...),
            SNRCalculator(),
            PrintMetric("snr", label="after PCA"),   # → "  [after PCA] snr=14.201"
        ])
    """

    name = "print_metric"
    description = "Print evaluation metric values for debugging"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = False
    modifies_raw = False
    parallel_safe = False

    def __init__(self, *metric_names: str, label: str | None = None):
        self._metric_names = metric_names
        self._label = label
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- COMPUTE ---
        parts = []
        for name in self._metric_names:
            val = context.get_metric(name)
            if isinstance(val, float):
                parts.append(f"{name}={val:.3f}")
            elif val is not None:
                parts.append(f"{name}={val}")
            else:
                parts.append(f"{name}=N/A")

        prefix = f"[{self._label}] " if self._label else ""
        message = "{}{}".format(prefix, ", ".join(parts))

        # --- LOG ---
        logger.info("{}", message)
        print(f"  {message}")

        # --- RETURN ---
        return context


@register_processor
class RawTransform(Processor):
    """Apply an arbitrary callable to the Raw object.

    A lighter-weight alternative to ``LambdaProcessor`` when only the Raw
    object needs to be modified.  The callable receives the **current** Raw
    object and must return a *new* (or modified copy of a) Raw object.

    Parameters
    ----------
    name : str
        Human-readable label shown in pipeline logs and progress.
    func : callable
        ``Callable[[mne.io.Raw], mne.io.Raw]`` — receives the current Raw
        object, must return a (possibly new) Raw object.

    Examples
    --------
    ::

        # Drop bad channels inline
        RawTransform("drop_bad", lambda raw: raw.copy().pick_channels(
            [ch for ch in raw.ch_names if ch not in ["EKG", "EMG"]]
        ))

        # Set average reference
        RawTransform("set_eeg_ref", lambda raw: raw.copy().set_eeg_reference("average"))
    """

    name = "raw_transform"
    description = "Apply a callable transform to the Raw object"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = False

    def __init__(self, name: str, func: Callable):
        self.name = name
        self._func = func
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- LOG ---
        logger.info("Applying raw transform: {}", self.name)

        # --- COMPUTE + RETURN ---
        new_raw = self._func(context.get_raw())
        return context.with_raw(new_raw)
