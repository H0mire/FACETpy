"""
Simple raw-transform processors.

Contains small, focused processors for common in-pipeline data manipulations
that don't fit neatly into filtering, resampling, or trigger handling.
"""

from typing import Callable, List, Optional, Union

from loguru import logger

from ..core import Processor, ProcessingContext, register_processor


@register_processor
class Crop(Processor):
    """
    Crop the Raw recording to a time interval.

    A concise alternative to ``LambdaProcessor`` for the common pattern of
    restricting the recording to a specific window before processing.

    Example::

        Crop(tmin=0, tmax=162)

        # Equivalent LambdaProcessor form (no longer necessary):
        # LambdaProcessor(
        #     name="crop",
        #     func=lambda ctx: ctx.with_raw(ctx.get_raw().copy().crop(tmin=0, tmax=162))
        # )
    """

    name = "crop"
    description = "Crop Raw recording to a time interval"
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(
        self,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
    ):
        """
        Args:
            tmin: Start time in seconds (``None`` keeps the original start).
            tmax: End time in seconds (``None`` keeps the original end).
        """
        self.tmin = tmin
        self.tmax = tmax
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        kwargs = {}
        if self.tmin is not None:
            kwargs['tmin'] = self.tmin
        if self.tmax is not None:
            kwargs['tmax'] = self.tmax
        logger.info(f"Cropping raw: tmin={self.tmin}, tmax={self.tmax}")
        raw.crop(**kwargs)
        return context.with_raw(raw)


@register_processor
class PickChannels(Processor):
    """
    Keep only the specified channels or channel types.

    A named, reusable alternative to the common ``lambda ctx: ctx.with_raw(
    ctx.get_raw().copy().pick(...))`` pattern.

    Example::

        # Keep only EEG and stimulus channels
        PickChannels(picks=["eeg", "stim"])

        # Keep specific channels by name
        PickChannels(picks=["Fp1", "Fp2", "Fz"])
    """

    name = "pick_channels"
    description = "Keep only the specified channels or channel types"
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(
        self,
        picks: Union[str, List[str]],
        on_missing: str = "ignore",
    ):
        """
        Args:
            picks: Channel type (``"eeg"``, ``"stim"``, …) or list of channel
                names / types accepted by :meth:`mne.io.Raw.pick`.
            on_missing: Passed to MNE.  ``"ignore"`` (default) silently skips
                channels that are absent from the recording.
        """
        self.picks = picks
        self.on_missing = on_missing
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        logger.info(f"Picking channels: {self.picks}")
        raw = context.get_raw().copy().pick(picks=self.picks, verbose=False)
        return context.with_raw(raw)


@register_processor
class DropChannels(Processor):
    """
    Remove named channels from the recording.

    A named, reusable alternative to the ``lambda ctx: ...drop_channels(...)``
    pattern commonly seen in inline pipeline steps.

    Example::

        # Drop typical non-EEG channels that may be present in EDF files
        DropChannels(channels=["EKG", "EMG", "EOG", "ECG"])
    """

    name = "drop_channels"
    description = "Remove named channels from the recording"
    requires_raw = True
    modifies_raw = True
    parallel_safe = True

    def __init__(self, channels: List[str], on_missing: str = "ignore"):
        """
        Args:
            channels: List of channel names to remove.
            on_missing: ``"ignore"`` (default) skips absent names silently;
                ``"raise"`` raises an error when a channel is not found.
        """
        self.channels = channels
        self.on_missing = on_missing
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        if self.on_missing == "ignore":
            to_drop = [ch for ch in self.channels if ch in raw.ch_names]
        else:
            to_drop = self.channels
        if to_drop:
            logger.info(f"Dropping channels: {to_drop}")
            raw.drop_channels(to_drop)
        return context.with_raw(raw)


@register_processor
class PrintMetric(Processor):
    """
    Print one or more evaluation metric values — useful for debugging pipelines.

    Inserts a transparent logging step that reads from the shared metrics dict
    populated by evaluation processors (e.g. :class:`~facet.evaluation.SNRCalculator`).
    The context is returned unchanged.

    Example::

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
    requires_raw = False
    modifies_raw = False
    parallel_safe = False

    def __init__(self, *metric_names: str, label: Optional[str] = None):
        """
        Args:
            *metric_names: One or more metric names to print
                (e.g. ``'snr'``, ``'rms_ratio'``).
            label: Optional prefix shown in brackets, e.g. ``"after PCA"``.
        """
        self._metric_names = metric_names
        self._label = label
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
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
        logger.info(f"{prefix}{', '.join(parts)}")
        print(f"  {prefix}{', '.join(parts)}")
        return context


@register_processor
class RawTransform(Processor):
    """
    Apply an arbitrary callable to the Raw object.

    A lighter-weight alternative to ``LambdaProcessor`` when only the Raw
    object needs to be modified.  The callable receives the **current** Raw
    object and must return a *new* (or modified copy of a) Raw object.

    Example::

        # Drop bad channels inline
        RawTransform("drop_bad", lambda raw: raw.copy().pick_channels(
            [ch for ch in raw.ch_names if ch not in ["EKG", "EMG"]]
        ))

        # Set average reference
        RawTransform("set_eeg_ref", lambda raw: raw.copy().set_eeg_reference("average"))
    """

    name = "raw_transform"
    description = "Apply a callable transform to the Raw object"
    requires_raw = True
    modifies_raw = True
    parallel_safe = False

    def __init__(self, name: str, func: Callable):
        """
        Args:
            name: Human-readable label shown in pipeline logs and progress.
            func: ``Callable[[mne.io.Raw], mne.io.Raw]`` — receives the current
                  Raw object, must return a (possibly new) Raw object.
        """
        self.name = name
        self._func = func
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        logger.info(f"Applying raw transform: {self.name}")
        new_raw = self._func(context.get_raw())
        return context.with_raw(new_raw)
