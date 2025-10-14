"""
Resampling Processors Module

This module contains processors for resampling EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional
import mne
from loguru import logger
import numpy as np

from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError


@register_processor
class Resample(Processor):
    """
    Generic resampling processor.

    Example:
        resample = Resample(sfreq=1000.0)
        context = resample.execute(context)
    """

    name = "resample"
    description = "Resample EEG data to a new sampling frequency"
    parallel_safe = True
    parallelize_by_channels = True

    def __init__(
        self,
        sfreq: float,
        npad: str = 'auto',
        window: str = 'boxcar',
        n_jobs: int = 1,
        verbose: bool = False
    ):
        """
        Initialize resampler.

        Args:
            sfreq: Target sampling frequency in Hz
            npad: Amount to pad the start and end of the data
            window: Window to use for resampling
            n_jobs: Number of parallel jobs for resampling
            verbose: Verbose output
        """
        self.sfreq = sfreq
        self.npad = npad
        self.window = window
        self.n_jobs = n_jobs
        self.verbose = verbose
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Resample data."""
        raw = context.get_raw().copy()
        old_sfreq = raw.info['sfreq']

        logger.info(f"Resampling from {old_sfreq}Hz to {self.sfreq}Hz")

        # Resample raw data
        raw.resample(
            sfreq=self.sfreq,
            npad=self.npad,
            window=self.window,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        # Update metadata
        new_metadata = context.metadata.copy()

        # Update triggers if they exist
        if context.has_triggers():
            triggers = context.get_triggers()
            resampling_factor = self.sfreq / old_sfreq
            new_triggers = np.array([
                int(trigger * resampling_factor) for trigger in triggers
            ])
            new_metadata.triggers = new_triggers
            logger.debug(f"Updated {len(new_triggers)} trigger positions")

        # Create new context
        new_context = context.with_raw(raw)
        new_context._metadata = new_metadata

        # Resample estimated noise if it exists
        if context.has_estimated_noise():
            noise = context.get_estimated_noise().copy()
            noise_raw = context.get_raw_original().copy()
            noise_raw._data = noise
            noise_raw.resample(
                sfreq=self.sfreq,
                npad=self.npad,
                window=self.window,
                n_jobs=self.n_jobs,
                verbose=False
            )
            new_context.set_estimated_noise(noise_raw._data)

        return new_context


@register_processor
class UpSample(Resample):
    """
    Upsample processor.

    Increases sampling frequency by a factor.

    Example:
        upsample = UpSample(factor=10)
        context = upsample.execute(context)
    """

    name = "upsample"
    description = "Upsample EEG data by a factor"

    def __init__(
        self,
        factor: int = 10,
        npad: str = 'auto',
        window: str = 'boxcar',
        n_jobs: int = 1,
        verbose: bool = False
    ):
        """
        Initialize upsampler.

        Args:
            factor: Upsampling factor (e.g., 10 means 10x increase)
            npad: Amount to pad the start and end of the data
            window: Window to use for resampling
            n_jobs: Number of parallel jobs
            verbose: Verbose output
        """
        self.factor = factor
        # Initialize parent with a dummy sfreq (will be overridden in process)
        super().__init__(
            sfreq=1.0,  # Dummy value, set properly in process()
            npad=npad,
            window=window,
            n_jobs=n_jobs,
            verbose=verbose
        )

    def _get_parameters(self):
        """Override to include factor instead of sfreq."""
        return {
            'factor': self.factor,
            'npad': self.npad,
            'window': self.window,
            'n_jobs': self.n_jobs
        }

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Upsample data."""
        raw = context.get_raw()
        old_sfreq = raw.info['sfreq']
        target_sfreq = old_sfreq * self.factor

        # Temporarily set sfreq for parent class
        self.sfreq = target_sfreq

        logger.info(f"Upsampling by factor {self.factor} ({old_sfreq}Hz -> {target_sfreq}Hz)")

        # Call parent process method
        return super().process(context)


@register_processor
class DownSample(Resample):
    """
    Downsample processor.

    Decreases sampling frequency by a factor.

    Example:
        downsample = DownSample(factor=10)
        context = downsample.execute(context)
    """

    name = "downsample"
    description = "Downsample EEG data by a factor"

    def __init__(
        self,
        factor: int = 10,
        npad: str = 'auto',
        window: str = 'boxcar',
        n_jobs: int = 1,
        verbose: bool = False
    ):
        """
        Initialize downsampler.

        Args:
            factor: Downsampling factor (e.g., 10 means 10x decrease)
            npad: Amount to pad the start and end of the data
            window: Window to use for resampling
            n_jobs: Number of parallel jobs
            verbose: Verbose output
        """
        self.factor = factor
        # Initialize parent with a dummy sfreq (will be overridden in process)
        super().__init__(
            sfreq=1.0,  # Dummy value, set properly in process()
            npad=npad,
            window=window,
            n_jobs=n_jobs,
            verbose=verbose
        )

    def _get_parameters(self):
        """Override to include factor instead of sfreq."""
        return {
            'factor': self.factor,
            'npad': self.npad,
            'window': self.window,
            'n_jobs': self.n_jobs
        }

    def validate(self, context: ProcessingContext) -> None:
        """Validate that downsampling is possible."""
        super().validate(context)
        raw = context.get_raw()
        target_sfreq = raw.info['sfreq'] / self.factor
        if target_sfreq < 1:
            raise ProcessorValidationError(
                f"Downsampling factor {self.factor} would result in "
                f"sampling frequency < 1Hz"
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Downsample data."""
        raw = context.get_raw()
        old_sfreq = raw.info['sfreq']
        target_sfreq = old_sfreq / self.factor

        # Temporarily set sfreq for parent class
        self.sfreq = target_sfreq

        logger.info(f"Downsampling by factor {self.factor} ({old_sfreq}Hz -> {target_sfreq}Hz)")

        # Call parent process method
        return super().process(context)
