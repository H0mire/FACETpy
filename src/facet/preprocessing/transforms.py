"""
Simple raw-transform processors.

Contains small, focused processors for common in-pipeline data manipulations
that don't fit neatly into filtering, resampling, or trigger handling.
"""

from typing import Callable, Optional

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
