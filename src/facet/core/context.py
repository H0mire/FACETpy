"""
Processing Context Module

This module defines the ProcessingContext class which wraps MNE Raw objects
and provides metadata storage, history tracking, and immutable operations.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
import mne
from loguru import logger
from facet.logging_config import suppress_stdout


@dataclass
class ProcessingMetadata:
    """Metadata associated with processing context."""

    triggers: Optional[np.ndarray] = None
    trigger_regex: Optional[str] = None
    artifact_to_trigger_offset: float = 0.0
    acq_start_sample: Optional[int] = None
    acq_end_sample: Optional[int] = None
    pre_trigger_samples: Optional[int] = None
    post_trigger_samples: Optional[int] = None
    upsampling_factor: int = 1
    artifact_length: Optional[int] = None
    slices_per_volume: Optional[int] = None
    volume_gaps: Optional[bool] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> 'ProcessingMetadata':
        """Create a deep copy of metadata."""
        return ProcessingMetadata(
            triggers=self.triggers.copy() if self.triggers is not None else None,
            trigger_regex=self.trigger_regex,
            artifact_to_trigger_offset=self.artifact_to_trigger_offset,
            acq_start_sample=self.acq_start_sample,
            acq_end_sample=self.acq_end_sample,
            pre_trigger_samples=self.pre_trigger_samples,
            post_trigger_samples=self.post_trigger_samples,
            upsampling_factor=self.upsampling_factor,
            artifact_length=self.artifact_length,
            slices_per_volume=self.slices_per_volume,
            volume_gaps=self.volume_gaps,
            custom=deepcopy(self.custom)
        )


@dataclass
class ProcessingStep:
    """Record of a processing step."""

    name: str
    processor_type: str
    parameters: Dict[str, Any]
    timestamp: float


class ProcessingContext:
    """
    Processing context that wraps MNE Raw objects and provides metadata.

    This class serves as the primary data container passed between processors.
    It provides:
    - Access to current and original MNE Raw objects
    - Metadata storage (triggers, parameters, etc.)
    - Processing history tracking
    - Estimated noise tracking
    - Immutable operations (processors return new contexts)

    Attributes:
        _raw: Current processed MNE Raw object
        _raw_original: Original unprocessed MNE Raw object
        _metadata: Processing metadata
        _history: List of processing steps
        _estimated_noise: Accumulated noise estimates
        _cache: Cache for computed values
    """

    def __init__(
        self,
        raw: mne.io.Raw,
        raw_original: Optional[mne.io.Raw] = None,
        metadata: Optional[ProcessingMetadata] = None
    ):
        """
        Initialize processing context.

        Args:
            raw: MNE Raw object
            raw_original: Original Raw object (if None, copies raw)
            metadata: Processing metadata (if None, creates empty)
        """
        self._raw = raw
        self._raw_original = raw_original if raw_original is not None else raw.copy()
        self._metadata = metadata if metadata is not None else ProcessingMetadata()
        self._history: List[ProcessingStep] = []
        self._estimated_noise: Optional[np.ndarray] = None
        self._cache: Dict[str, Any] = {}

    # =========================================================================
    # Raw Data Access
    # =========================================================================

    def get_raw(self) -> mne.io.Raw:
        """Get current processed Raw object."""
        return self._raw

    def get_raw_original(self) -> mne.io.Raw:
        """Get original unprocessed Raw object."""
        return self._raw_original

    def has_raw(self) -> bool:
        """Check if Raw object exists."""
        return self._raw is not None

    def _call_raw_get_data(
        self,
        picks: Optional[Any] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Call Raw.get_data while gracefully handling removed keywords.

        Some lightweight MNE containers (e.g., RawArray created during chunk
        execution) do not accept the ``copy`` keyword. We optimistically pass
        it to avoid extra allocations and fall back to the default behaviour if
        the object rejects it.
        """
        try:
            return self._raw.get_data(picks=picks, **kwargs)
        except TypeError as exc:
            if 'copy' in kwargs and "unexpected keyword argument 'copy'" in str(exc):
                logger.debug("Raw.get_data() rejected 'copy'; retrying without it")
                safe_kwargs = dict(kwargs)
                safe_kwargs.pop('copy', None)
                return self._raw.get_data(picks=picks, **safe_kwargs)
            raise

    def get_data(self, picks: Optional[Any] = None, **kwargs) -> np.ndarray:
        """
        Get data from Raw object.

        Args:
            picks: Channel picks (MNE format)
            **kwargs: Additional arguments for Raw.get_data(copy=False)

        Returns:
            Data array
        """
        return self._call_raw_get_data(picks=picks, **kwargs)

    # =========================================================================
    # Metadata Access
    # =========================================================================

    @property
    def metadata(self) -> ProcessingMetadata:
        """Get processing metadata."""
        return self._metadata

    def get_triggers(self) -> Optional[np.ndarray]:
        """Get trigger positions."""
        return self._metadata.triggers

    def has_triggers(self) -> bool:
        """Check if triggers exist."""
        return self._metadata.triggers is not None and len(self._metadata.triggers) > 0

    def get_artifact_length(self) -> Optional[int]:
        """Get artifact length in samples."""
        return self._metadata.artifact_length

    def get_sfreq(self) -> float:
        """Get sampling frequency."""
        return self._raw.info['sfreq']

    def get_n_channels(self) -> int:
        """Get number of channels."""
        return len(self._raw.ch_names)

    def get_channel_names(self) -> List[str]:
        """Get channel names."""
        return self._raw.ch_names

    # =========================================================================
    # Noise Tracking
    # =========================================================================

    def get_estimated_noise(self) -> Optional[np.ndarray]:
        """Get accumulated noise estimates."""
        return self._estimated_noise

    def has_estimated_noise(self) -> bool:
        """Check if noise estimates exist."""
        return self._estimated_noise is not None

    def set_estimated_noise(self, noise: np.ndarray) -> None:
        """Set estimated noise (mutable operation)."""
        self._estimated_noise = noise

    def accumulate_noise(self, noise: np.ndarray) -> None:
        """Add to accumulated noise estimates."""
        if self._estimated_noise is None:
            self._estimated_noise = noise.copy()
        else:
            self._estimated_noise += noise

    # =========================================================================
    # History Tracking
    # =========================================================================

    def get_history(self) -> List[ProcessingStep]:
        """Get processing history."""
        return self._history.copy()

    def add_history_entry(
        self,
        name: str,
        processor_type: str,
        parameters: Dict[str, Any]
    ) -> None:
        """Add entry to processing history."""
        import time
        step = ProcessingStep(
            name=name,
            processor_type=processor_type,
            parameters=parameters,
            timestamp=time.time()
        )
        self._history.append(step)
        logger.debug(f"Added processing step: {name}")

    # =========================================================================
    # Cache Management
    # =========================================================================

    def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)

    def cache_set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = value

    def cache_has(self, key: str) -> bool:
        """Check if cache has key."""
        return key in self._cache

    def cache_clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    # =========================================================================
    # Immutable Operations (Return New Context)
    # =========================================================================

    def with_raw(
        self,
        raw: mne.io.Raw,
        copy_metadata: bool = True
    ) -> 'ProcessingContext':
        """
        Create new context with updated Raw object.

        Args:
            raw: New Raw object
            copy_metadata: Whether to copy metadata from current context

        Returns:
            New ProcessingContext
        """
        metadata = self._metadata.copy() if copy_metadata else ProcessingMetadata()

        new_ctx = ProcessingContext(
            raw=raw,
            raw_original=self._raw_original,
            metadata=metadata
        )
        new_ctx._history = self._history.copy()
        new_ctx._estimated_noise = self._estimated_noise.copy() if self._estimated_noise is not None else None
        # Don't copy cache (it may be invalidated)
        return new_ctx

    def with_metadata(self, metadata: ProcessingMetadata) -> 'ProcessingContext':
        """
        Create new context with updated metadata.

        Args:
            metadata: New metadata

        Returns:
            New ProcessingContext
        """
        new_ctx = ProcessingContext(
            raw=self._raw,
            raw_original=self._raw_original,
            metadata=metadata
        )
        new_ctx._history = self._history.copy()
        new_ctx._estimated_noise = self._estimated_noise.copy() if self._estimated_noise is not None else None
        return new_ctx

    def with_triggers(self, triggers: np.ndarray) -> 'ProcessingContext':
        """
        Create new context with updated triggers.

        Args:
            triggers: New trigger array

        Returns:
            New ProcessingContext
        """
        new_metadata = self._metadata.copy()
        new_metadata.triggers = triggers
        return self.with_metadata(new_metadata)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self, deep: bool = False) -> 'ProcessingContext':
        """
        Create a copy of the context.

        Args:
            deep: If True, deep copy Raw object (expensive)

        Returns:
            Copied ProcessingContext
        """
        raw_copy = self._raw.copy() if deep else self._raw
        return self.with_raw(raw_copy, copy_metadata=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize context to dictionary (for multiprocessing).

        Returns:
            Dictionary representation
        """
        return {
            'raw_data': self._call_raw_get_data(copy=False),
            'raw_info': self._raw.info,
            'metadata': {
                'triggers': self._metadata.triggers,
                'trigger_regex': self._metadata.trigger_regex,
                'artifact_to_trigger_offset': self._metadata.artifact_to_trigger_offset,
                'acq_start_sample': self._metadata.acq_start_sample,
                'acq_end_sample': self._metadata.acq_end_sample,
                'pre_trigger_samples': self._metadata.pre_trigger_samples,
                'post_trigger_samples': self._metadata.post_trigger_samples,
                'upsampling_factor': self._metadata.upsampling_factor,
                'artifact_length': self._metadata.artifact_length,
                'slices_per_volume': self._metadata.slices_per_volume,
                'volume_gaps': self._metadata.volume_gaps,
                'custom': self._metadata.custom
            },
            'estimated_noise': self._estimated_noise
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingContext':
        """
        Deserialize context from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ProcessingContext instance
        """
        # Reconstruct Raw object (suppress MNE's verbose print output)
        with suppress_stdout():
            raw = mne.io.RawArray(data['raw_data'], data['raw_info'])

        # Reconstruct metadata
        metadata = ProcessingMetadata(
            triggers=data['metadata']['triggers'],
            trigger_regex=data['metadata']['trigger_regex'],
            artifact_to_trigger_offset=data['metadata']['artifact_to_trigger_offset'],
            acq_start_sample=data['metadata'].get('acq_start_sample'),
            acq_end_sample=data['metadata'].get('acq_end_sample'),
            pre_trigger_samples=data['metadata'].get('pre_trigger_samples'),
            post_trigger_samples=data['metadata'].get('post_trigger_samples'),
            upsampling_factor=data['metadata']['upsampling_factor'],
            artifact_length=data['metadata']['artifact_length'],
            slices_per_volume=data['metadata'].get('slices_per_volume'),
            volume_gaps=data['metadata']['volume_gaps'],
            custom=data['metadata']['custom']
        )

        ctx = cls(raw=raw, metadata=metadata)
        ctx._estimated_noise = data['estimated_noise']
        return ctx

    def __repr__(self) -> str:
        """String representation."""
        n_channels = len(self._raw.ch_names)
        n_times = self._raw.n_times
        sfreq = self._raw.info['sfreq']
        n_triggers = len(self._metadata.triggers) if self._metadata.triggers is not None else 0
        return (
            f"ProcessingContext(n_channels={n_channels}, n_times={n_times}, "
            f"sfreq={sfreq}Hz, n_triggers={n_triggers}, "
            f"n_steps={len(self._history)})"
        )
