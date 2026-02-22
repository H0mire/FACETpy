"""
Simple raw-transform processors.

Contains small, focused processors for common in-pipeline data manipulations
that don't fit neatly into filtering, resampling, or trigger handling.
"""

from collections.abc import Callable

from loguru import logger

from ..core import ProcessingContext, Processor, register_processor


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

    Examples
    --------
    ::

        Crop(tmin=0, tmax=162)
    """

    name = "crop"
    description = "Crop Raw recording to a time interval"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
    ):
        self.tmin = tmin
        self.tmax = tmax
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()

        # --- LOG ---
        logger.info("Cropping raw: tmin={}, tmax={}", self.tmin, self.tmax)

        # --- COMPUTE ---
        kwargs = {}
        if self.tmin is not None:
            kwargs["tmin"] = self.tmin
        if self.tmax is not None:
            kwargs["tmax"] = self.tmax
        raw.crop(**kwargs)

        # --- RETURN ---
        return context.with_raw(raw)


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
