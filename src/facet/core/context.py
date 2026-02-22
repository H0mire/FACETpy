"""
Processing Context Module

This module defines the ProcessingContext class which wraps MNE Raw objects
and provides metadata storage, history tracking, and immutable operations.

Author: FACETpy Team
Date: 2025-01-12
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import mne
import numpy as np
from loguru import logger

from facet.logging_config import suppress_stdout


@dataclass
class ProcessingMetadata:
    """Metadata associated with processing context."""

    triggers: np.ndarray | None = None
    trigger_regex: str | None = None
    artifact_to_trigger_offset: float = 0.0
    acq_start_sample: int | None = None
    acq_end_sample: int | None = None
    pre_trigger_samples: int | None = None
    post_trigger_samples: int | None = None
    upsampling_factor: int = 10
    artifact_length: int | None = None
    slices_per_volume: int | None = None
    volume_gaps: bool = False
    custom: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ProcessingMetadata":
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
            custom=deepcopy(self.custom),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "triggers": self.triggers.tolist() if self.triggers is not None else None,
            "trigger_regex": self.trigger_regex,
            "artifact_to_trigger_offset": self.artifact_to_trigger_offset,
            "acq_start_sample": self.acq_start_sample,
            "acq_end_sample": self.acq_end_sample,
            "pre_trigger_samples": self.pre_trigger_samples,
            "post_trigger_samples": self.post_trigger_samples,
            "upsampling_factor": self.upsampling_factor,
            "artifact_length": self.artifact_length,
            "slices_per_volume": self.slices_per_volume,
            "volume_gaps": self.volume_gaps,
            "custom": deepcopy(self.custom),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingMetadata":
        """Deserialize metadata from dictionary."""
        triggers = data.get("triggers")
        if triggers is not None:
            triggers = np.array(triggers)
        return cls(
            triggers=triggers,
            trigger_regex=data.get("trigger_regex"),
            artifact_to_trigger_offset=data.get("artifact_to_trigger_offset", 0.0),
            acq_start_sample=data.get("acq_start_sample"),
            acq_end_sample=data.get("acq_end_sample"),
            pre_trigger_samples=data.get("pre_trigger_samples"),
            post_trigger_samples=data.get("post_trigger_samples"),
            upsampling_factor=data.get("upsampling_factor", 10),
            artifact_length=data.get("artifact_length"),
            slices_per_volume=data.get("slices_per_volume"),
            volume_gaps=data.get("volume_gaps", False),
            custom=deepcopy(data.get("custom", {})),
        )


@dataclass
class ProcessingStep:
    """Record of a processing step."""

    name: str
    processor_type: str
    parameters: dict[str, Any]
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
        self, raw: mne.io.Raw, raw_original: mne.io.Raw | None = None, metadata: ProcessingMetadata | None = None
    ):
        """
        Initialize processing context.

        Args:
            raw: MNE Raw object
            raw_original: Original Raw object (if None, copies raw)
            metadata: Processing metadata (if None, creates empty)
        """
        self._raw = raw
        self._raw_original = raw_original if raw_original is not None else raw
        self._metadata = metadata if metadata is not None else ProcessingMetadata()
        self._history: list[ProcessingStep] = []
        self._estimated_noise: np.ndarray | None = None
        self._cache: dict[str, Any] = {}

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

    def _call_raw_get_data(self, picks: Any | None = None, **kwargs) -> np.ndarray:
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
            if "copy" in kwargs and "unexpected keyword argument 'copy'" in str(exc):
                logger.debug("Raw.get_data() rejected 'copy'; retrying without it")
                safe_kwargs = dict(kwargs)
                safe_kwargs.pop("copy", None)
                return self._raw.get_data(picks=picks, **safe_kwargs)
            raise

    def get_data(self, picks: Any | None = None, **kwargs) -> np.ndarray:
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

    def get_triggers(self) -> np.ndarray | None:
        """Get trigger positions."""
        return self._metadata.triggers

    def has_triggers(self) -> bool:
        """Check if triggers exist."""
        return self._metadata.triggers is not None and len(self._metadata.triggers) > 0

    def get_artifact_length(self) -> int | None:
        """Get artifact length in samples."""
        return self._metadata.artifact_length

    def get_sfreq(self) -> float:
        """Get sampling frequency."""
        return self._raw.info["sfreq"]

    def get_n_channels(self) -> int:
        """Get number of channels."""
        return len(self._raw.ch_names)

    def get_channel_names(self) -> list[str]:
        """Get channel names."""
        return self._raw.ch_names

    def get_metric(self, name: str, default=None):
        """
        Return a single evaluation metric stored in the context.

        Shortcut for the common pattern::

            ctx.metadata.custom.get('metrics', {}).get(name, default)

        Typically used inside a :class:`~facet.core.ConditionalProcessor`
        condition function after an evaluation step has run.

        Args:
            name: Metric name (e.g. ``'snr'``, ``'rms_ratio'``).
            default: Value returned when the metric is absent.

        Example::

            def needs_extra_correction(ctx):
                return ctx.get_metric('snr', float('inf')) < 10
        """
        return self._metadata.custom.get("metrics", {}).get(name, default)

    # =========================================================================
    # Noise Tracking
    # =========================================================================

    def get_estimated_noise(self) -> np.ndarray | None:
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

    def get_history(self) -> list[ProcessingStep]:
        """Get processing history."""
        return self._history.copy()

    def add_history_entry(
        self,
        name: str | None = None,
        processor_type: str = "",
        parameters: dict[str, Any] | None = None,
        *,
        processor_name: str | None = None,
    ) -> None:
        """Add entry to processing history."""
        import time

        resolved_name = processor_name if processor_name is not None else (name or "")
        step = ProcessingStep(
            name=resolved_name, processor_type=processor_type, parameters=parameters or {}, timestamp=time.time()
        )
        self._history.append(step)
        logger.debug(f"Added processing step: {resolved_name}")

    # =========================================================================
    # Cache Management
    # =========================================================================

    def cache_get(self, key: str) -> Any | None:
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

    def with_raw(self, raw: mne.io.Raw, copy_metadata: bool = True) -> "ProcessingContext":
        """
        Create new context with updated Raw object.

        Args:
            raw: New Raw object
            copy_metadata: Whether to copy metadata from current context

        Returns:
            New ProcessingContext
        """
        metadata = self._metadata.copy() if copy_metadata else ProcessingMetadata()

        new_ctx = ProcessingContext(raw=raw, raw_original=self._raw_original, metadata=metadata)
        new_ctx._history = self._history.copy()
        new_ctx._estimated_noise = self._estimated_noise.copy() if self._estimated_noise is not None else None
        # Don't copy cache (it may be invalidated)
        return new_ctx

    def with_metadata(self, metadata: ProcessingMetadata) -> "ProcessingContext":
        """
        Create new context with updated metadata.

        Args:
            metadata: New metadata

        Returns:
            New ProcessingContext
        """
        new_ctx = ProcessingContext(raw=self._raw, raw_original=self._raw_original, metadata=metadata)
        new_ctx._history = self._history.copy()
        new_ctx._estimated_noise = self._estimated_noise.copy() if self._estimated_noise is not None else None
        return new_ctx

    def with_triggers(self, triggers: np.ndarray) -> "ProcessingContext":
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

    def copy(self, deep: bool = False) -> "ProcessingContext":
        """
        Create a copy of the context.

        Args:
            deep: If True, deep copy Raw object (expensive)

        Returns:
            Copied ProcessingContext
        """
        raw_copy = self._raw.copy() if deep else self._raw
        return self.with_raw(raw_copy, copy_metadata=True)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize context to dictionary (for multiprocessing).

        Returns:
            Dictionary representation
        """
        return {
            "raw": self._raw,
            "raw_data": self._call_raw_get_data(copy=False),
            "raw_info": self._raw.info,
            "metadata": self._metadata.to_dict(),
            "history": [
                {
                    "name": step.name,
                    "processor_type": step.processor_type,
                    "parameters": step.parameters,
                    "timestamp": step.timestamp,
                }
                for step in self._history
            ],
            "estimated_noise": self._estimated_noise,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingContext":
        """
        Deserialize context from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ProcessingContext instance
        """
        # suppress_stdout: RawArray creation prints verbose MNE messages
        with suppress_stdout():
            raw = mne.io.RawArray(data["raw_data"], data["raw_info"])

        metadata = ProcessingMetadata.from_dict(data["metadata"])

        ctx = cls(raw=raw, metadata=metadata)
        ctx._estimated_noise = data["estimated_noise"]
        return ctx

    def __or__(self, other) -> "ProcessingContext":
        """
        Apply a processor or callable to this context using the pipe operator.

        Enables a clean chaining syntax outside of a ``Pipeline``::

            ctx = ProcessingContext(raw)
            result = (
                ctx
                | HighPassFilter(1.0)
                | UpSample(10)
                | TriggerDetector(r"\\b1\\b")
                | AASCorrection()
            )
            filtered_raw = result.get_raw()

        Args:
            other: A :class:`~facet.core.Processor` instance **or** any
                ``Callable[[ProcessingContext], ProcessingContext]``.

        Returns:
            New :class:`ProcessingContext` produced by applying *other*.
        """
        from .processor import Processor

        if isinstance(other, Processor):
            return other.execute(self)
        elif callable(other):
            return other(self)
        return NotImplemented

    def __repr__(self) -> str:
        """String representation."""
        n_channels = len(self._raw.ch_names)
        n_times = self._raw.n_times
        sfreq = self._raw.info["sfreq"]
        n_triggers = len(self._metadata.triggers) if self._metadata.triggers is not None else 0
        return (
            f"ProcessingContext(n_channels={n_channels}, n_times={n_times}, "
            f"sfreq={sfreq}Hz, n_triggers={n_triggers}, "
            f"n_steps={len(self._history)})"
        )
