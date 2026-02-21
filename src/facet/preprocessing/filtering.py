"""
Filtering Processors Module

This module contains processors for filtering EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, List, Union
import mne
from loguru import logger
import numpy as np

from ..core import Processor, ProcessingContext, register_processor


@register_processor
class Filter(Processor):
    """
    Generic filter processor.

    Applies MNE's filter method with specified parameters.

    Example:
        filter = Filter(l_freq=1.0, h_freq=70.0)
        context = filter.execute(context)
    """

    name = "filter"
    description = "Apply generic filter to EEG data"
    parallel_safe = True
    parallelize_by_channels = True

    def __init__(
        self,
        l_freq: Optional[float] = None,
        h_freq: Optional[float] = None,
        picks: Optional[Union[str, List[str]]] = None,
        filter_length: str = 'auto',
        method: str = 'fir',
        phase: str = 'zero',
        fir_window: str = 'hamming',
        verbose: bool = False
    ):
        """
        Initialize filter.

        Args:
            l_freq: Low cutoff frequency (None for no highpass)
            h_freq: High cutoff frequency (None for no lowpass)
            picks: Channels to filter (None for all)
            filter_length: Length of the FIR filter
            method: Filter method ('fir' or 'iir')
            phase: Phase of the filter ('zero', 'zero-double', 'minimum')
            fir_window: Window to use for FIR filter
            verbose: Verbose output
        """
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.picks = picks
        self.filter_length = filter_length
        self.method = method
        self.phase = phase
        self.fir_window = fir_window
        self.verbose = verbose
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()

        if self.l_freq and self.h_freq:
            filter_type = f"bandpass ({self.l_freq}-{self.h_freq}Hz)"
        elif self.l_freq:
            filter_type = f"highpass ({self.l_freq}Hz)"
        elif self.h_freq:
            filter_type = f"lowpass ({self.h_freq}Hz)"
        else:
            filter_type = "no filter"

        logger.info(f"Applying {filter_type}")

        raw.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            picks=self.picks,
            filter_length=self.filter_length,
            method=self.method,
            phase=self.phase,
            fir_window=self.fir_window,
            verbose=self.verbose
        )

        new_context = context.with_raw(raw)
        if context.has_estimated_noise():
            noise = context.get_estimated_noise().copy()
            noise_raw = raw.copy()
            noise_raw._data = noise
            noise_raw.filter(
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                picks=self.picks,
                filter_length=self.filter_length,
                method=self.method,
                phase=self.phase,
                fir_window=self.fir_window,
                verbose=False
            )
            new_context.set_estimated_noise(noise_raw._data)

        return new_context


@register_processor
class HighPassFilter(Filter):
    """
    Highpass filter processor.

    Example:
        filter = HighPassFilter(freq=1.0)
        context = filter.execute(context)
    """

    name = "highpass_filter"
    description = "Apply highpass filter to EEG data"

    def __init__(
        self,
        freq: float,
        picks: Optional[Union[str, List[str]]] = None,
        filter_length: str = 'auto',
        method: str = 'fir',
        phase: str = 'zero',
        fir_window: str = 'hamming'
    ):
        """
        Initialize highpass filter.

        Args:
            freq: Highpass cutoff frequency in Hz
            picks: Channels to filter (None for all)
            filter_length: Length of the FIR filter
            method: Filter method ('fir' or 'iir')
            phase: Phase of the filter
            fir_window: Window to use for FIR filter
        """
        super().__init__(
            l_freq=freq,
            h_freq=None,
            picks=picks,
            filter_length=filter_length,
            method=method,
            phase=phase,
            fir_window=fir_window
        )

    def _get_parameters(self):
        """Expose user-facing freq parameter for history/serialization."""
        return {
            'freq': self.l_freq,
            'picks': self.picks,
            'filter_length': self.filter_length,
            'method': self.method,
            'phase': self.phase,
            'fir_window': self.fir_window
        }


@register_processor
class LowPassFilter(Filter):
    """
    Lowpass filter processor.

    Example:
        filter = LowPassFilter(freq=70.0)
        context = filter.execute(context)
    """

    name = "lowpass_filter"
    description = "Apply lowpass filter to EEG data"

    def __init__(
        self,
        freq: float,
        picks: Optional[Union[str, List[str]]] = None,
        filter_length: str = 'auto',
        method: str = 'fir',
        phase: str = 'zero',
        fir_window: str = 'hamming'
    ):
        """
        Initialize lowpass filter.

        Args:
            freq: Lowpass cutoff frequency in Hz
            picks: Channels to filter (None for all)
            filter_length: Length of the FIR filter
            method: Filter method ('fir' or 'iir')
            phase: Phase of the filter
            fir_window: Window to use for FIR filter
        """
        super().__init__(
            l_freq=None,
            h_freq=freq,
            picks=picks,
            filter_length=filter_length,
            method=method,
            phase=phase,
            fir_window=fir_window
        )

    def _get_parameters(self):
        """Expose user-facing freq parameter for history/serialization."""
        return {
            'freq': self.h_freq,
            'picks': self.picks,
            'filter_length': self.filter_length,
            'method': self.method,
            'phase': self.phase,
            'fir_window': self.fir_window
        }


@register_processor
class BandPassFilter(Filter):
    """
    Bandpass filter processor.

    Example:
        filter = BandPassFilter(l_freq=1.0, h_freq=70.0)
        context = filter.execute(context)
    """

    name = "bandpass_filter"
    description = "Apply bandpass filter to EEG data"

    def __init__(
        self,
        l_freq: float,
        h_freq: float,
        picks: Optional[Union[str, List[str]]] = None,
        filter_length: str = 'auto',
        method: str = 'fir',
        phase: str = 'zero',
        fir_window: str = 'hamming'
    ):
        """
        Initialize bandpass filter.

        Args:
            l_freq: Low cutoff frequency in Hz
            h_freq: High cutoff frequency in Hz
            picks: Channels to filter (None for all)
            filter_length: Length of the FIR filter
            method: Filter method ('fir' or 'iir')
            phase: Phase of the filter
            fir_window: Window to use for FIR filter
        """
        super().__init__(
            l_freq=l_freq,
            h_freq=h_freq,
            picks=picks,
            filter_length=filter_length,
            method=method,
            phase=phase,
            fir_window=fir_window
        )


@register_processor
class NotchFilter(Processor):
    """
    Notch filter processor for removing line noise.

    Example:
        filter = NotchFilter(freqs=[50, 100])
        context = filter.execute(context)
    """

    name = "notch_filter"
    description = "Apply notch filter to remove line noise"
    parallel_safe = True
    parallelize_by_channels = True

    def __init__(
        self,
        freqs: Union[float, List[float]],
        picks: Optional[Union[str, List[str]]] = None,
        filter_length: str = 'auto',
        notch_widths: Optional[Union[float, List[float]]] = None,
        method: str = 'fir',
        phase: str = 'zero',
        fir_window: str = 'hamming'
    ):
        """
        Initialize notch filter.

        Args:
            freqs: Frequencies to notch filter (e.g., 50, 60 Hz for line noise)
            picks: Channels to filter (None for all)
            filter_length: Length of the FIR filter
            notch_widths: Width of each notch filter (None for default)
            method: Filter method ('fir' or 'iir')
            phase: Phase of the filter
            fir_window: Window to use for FIR filter
        """
        self.freqs = freqs if isinstance(freqs, list) else [freqs]
        self.picks = picks
        self.filter_length = filter_length
        self.notch_widths = notch_widths
        self.method = method
        self.phase = phase
        self.fir_window = fir_window
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()

        logger.info(f"Applying notch filter at {self.freqs}Hz")

        raw.notch_filter(
            freqs=self.freqs,
            picks=self.picks,
            filter_length=self.filter_length,
            notch_widths=self.notch_widths,
            method=self.method,
            phase=self.phase,
            fir_window=self.fir_window,
            verbose=False
        )

        new_context = context.with_raw(raw)
        if context.has_estimated_noise():
            noise = context.get_estimated_noise().copy()
            noise_raw = raw.copy()
            noise_raw._data = noise
            noise_raw.notch_filter(
                freqs=self.freqs,
                picks=self.picks,
                filter_length=self.filter_length,
                notch_widths=self.notch_widths,
                method=self.method,
                phase=self.phase,
                fir_window=self.fir_window,
                verbose=False
            )
            new_context.set_estimated_noise(noise_raw._data)

        return new_context
