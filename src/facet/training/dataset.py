"""Framework-agnostic EEG artifact dataset built from FACETpy ProcessingContexts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import mne
import numpy as np

from ..core import ProcessingContext

# ---------------------------------------------------------------------------
# Augmentation transforms (numpy, callable)
# ---------------------------------------------------------------------------


class TriggerJitter:
    """Randomly shift chunk start position by ±*max_jitter* samples.

    Applied at index-build time (offline).  For online augmentation use
    :class:`NoiseScaling` or :class:`SignFlip`.
    """

    def __init__(self, max_jitter: int = 5, seed: int = 0) -> None:
        self.max_jitter = max_jitter
        self._rng = np.random.default_rng(seed)

    def __call__(
        self, noisy: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        shift = int(self._rng.integers(-self.max_jitter, self.max_jitter + 1))
        if shift == 0:
            return noisy, target
        noisy = np.roll(noisy, shift, axis=-1)
        target = np.roll(target, shift, axis=-1)
        return noisy, target


class NoiseScaling:
    """Multiply signal by a random scalar drawn from *scale_range*.

    Example — add ±10 % amplitude variation::

        NoiseScaling(scale_range=(0.9, 1.1))
    """

    def __init__(
        self, scale_range: tuple[float, float] = (0.9, 1.1), seed: int = 0
    ) -> None:
        self.scale_range = scale_range
        self._rng = np.random.default_rng(seed)

    def __call__(
        self, noisy: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.scale_range
        if lo == hi == 1.0:
            return noisy, target
        scale = float(self._rng.uniform(lo, hi))
        return noisy * scale, target * scale


class ChannelDropout:
    """Zero-out each channel independently with probability *p*."""

    def __init__(self, p: float = 0.1, seed: int = 0) -> None:
        self.p = p
        self._rng = np.random.default_rng(seed)

    def __call__(
        self, noisy: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.p <= 0.0:
            return noisy, target
        mask = self._rng.random(noisy.shape[0]) > self.p  # (n_channels,)
        noisy = noisy * mask[:, np.newaxis]
        target = target * mask[:, np.newaxis]
        return noisy, target


class SignFlip:
    """Flip the polarity of the whole item with probability *p*."""

    def __init__(self, p: float = 0.5, seed: int = 0) -> None:
        self.p = p
        self._rng = np.random.default_rng(seed)

    def __call__(
        self, noisy: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.p <= 0.0:
            return noisy, target
        if self._rng.random() < self.p:
            return -noisy, -target
        return noisy, target


# ---------------------------------------------------------------------------
# Internal subset view
# ---------------------------------------------------------------------------


class _SubsetDataset:
    """Lightweight index-mapped view of an :class:`EEGArtifactDataset`."""

    def __init__(self, parent: EEGArtifactDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self._parent[self._indices[idx]]

    # Forward the framework-adapter helpers
    def to_torch(self) -> Any:
        return _TorchDatasetAdapter(self)

    def to_tf(self, batch_size: int = 16) -> Any:
        return _build_tf_dataset(self, batch_size)


class NPZContextArtifactDataset:
    """Dataset for context-based artifact prediction bundles stored as ``.npz``.

    The expected bundle is produced by
    ``examples/build_synthetic_spike_artifact_context_dataset.py``. Each item
    returns ``(noisy_context, artifact_center)`` by default, where the noisy
    input has shape ``(context_epochs, channels, epoch_samples)`` and the target
    has shape ``(channels, epoch_samples)``.
    """

    def __init__(
        self,
        path: str | Path,
        input_key: str = "noisy_context",
        target_key: str = "artifact_center",
        max_examples: int | None = None,
        demean_input: bool = False,
        demean_target: bool = False,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        with np.load(self.path, allow_pickle=True) as bundle:
            self.noisy = bundle[input_key].astype(np.float32, copy=False)
            self.target = bundle[target_key].astype(np.float32, copy=False)
            self.sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

        if self.noisy.shape[0] != self.target.shape[0]:
            raise ValueError("Context dataset input and target arrays must contain the same number of examples")
        if self.noisy.ndim != 4:
            raise ValueError("Context dataset input must have shape (examples, context_epochs, channels, samples)")
        if self.target.ndim != 3:
            raise ValueError("Context dataset target must have shape (examples, channels, samples)")

        if max_examples is not None:
            limit = max(0, min(int(max_examples), self.noisy.shape[0]))
            self.noisy = self.noisy[:limit]
            self.target = self.target[:limit]

        self.context_epochs = int(self.noisy.shape[1])
        self.n_channels = int(self.noisy.shape[2])
        self.epoch_samples = int(self.noisy.shape[3])
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact" if target_key == "artifact_center" else target_key
        self.trigger_aligned = True
        self.demean_input = demean_input
        self.demean_target = demean_target

    def __len__(self) -> int:
        return int(self.noisy.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        noisy = self.noisy[idx].copy()
        target = self.target[idx].copy()
        if self.demean_input:
            noisy -= noisy.mean(axis=-1, keepdims=True)
        if self.demean_target:
            target -= target.mean(axis=-1, keepdims=True)
        return noisy, target

    @property
    def n_chunks(self) -> int:
        return len(self)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.context_epochs, self.n_channels, self.epoch_samples)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (self.n_channels, self.epoch_samples)

    def train_val_split(
        self, val_ratio: float = 0.2, seed: int = 42
    ) -> tuple[_SubsetDataset, _SubsetDataset]:
        n = len(self)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n).tolist()
        n_val = max(1, int(n * val_ratio))
        val_idx = set(indices[:n_val])
        train_idx = [i for i in range(n) if i not in val_idx]
        val_idx_list = [i for i in range(n) if i in val_idx]
        return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx_list)

    def to_torch(self) -> Any:
        return _TorchDatasetAdapter(self)

    def to_tf(self, batch_size: int = 16) -> Any:
        return _build_tf_dataset(self, batch_size)


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------


class EEGArtifactDataset:
    """Framework-agnostic dataset of (noisy, target) EEG chunk pairs.

    Builds an in-memory index of chunk pairs from one or more
    :class:`~facet.core.ProcessingContext` objects.  Each item is a tuple
    ``(noisy, target)`` where both arrays have shape
    ``(n_channels, chunk_size)`` and dtype ``float32``.

    The dataset is framework-agnostic.  Use :meth:`to_torch` or
    :meth:`to_tf` to obtain a framework-native wrapper suitable for a
    ``DataLoader`` or ``tf.data`` pipeline.

    Parameters
    ----------
    contexts : ProcessingContext or list of ProcessingContext
        Source data.  Each context must have a valid raw EEG signal and,
        for supervised training, a stored ``raw_original``.
    chunk_size : int
        Samples per chunk.  Defaults to 1 250 (= 5 s at 250 Hz).
    target_type : {"clean", "artifact"}
        What the target array contains:

        * ``"clean"`` — the original clean signal from ``raw_original``
          (default; suitable for models that output a cleaned signal).
        * ``"artifact"`` — the gradient artifact estimate
          ``raw - raw_original`` (suitable for AAS-style models that
          output an artifact waveform to subtract).
    trigger_aligned : bool
        When ``True`` chunk boundaries are pinned to trigger onset samples.
        When ``False`` a sliding window with *overlap* is used instead.
    overlap : float
        Overlap ratio in ``[0, 1)`` for the sliding-window mode.
        Ignored when *trigger_aligned* is ``True``.
    transforms : list of callable, optional
        Augmentation transforms with signature
        ``(noisy, target) -> (noisy, target)``.  Applied at ``__getitem__``
        time (online).
    eeg_only : bool
        When ``True`` only EEG-typed channels are included.  When
        ``False`` all channels are kept (default: ``True``).

    Examples
    --------
    ::

        dataset = EEGArtifactDataset(context, chunk_size=1250)
        train_ds, val_ds = dataset.train_val_split(val_ratio=0.2)

        # PyTorch usage
        from torch.utils.data import DataLoader
        loader = DataLoader(train_ds.to_torch(), batch_size=16, shuffle=True)

        # TensorFlow usage
        tf_dataset = train_ds.to_tf(batch_size=16)
    """

    def __init__(
        self,
        contexts: ProcessingContext | list[ProcessingContext],
        chunk_size: int = 1250,
        target_type: str = "clean",
        trigger_aligned: bool = True,
        overlap: float = 0.0,
        transforms: list[Callable] | None = None,
        eeg_only: bool = True,
    ) -> None:
        if isinstance(contexts, ProcessingContext):
            contexts = [contexts]
        if not contexts:
            raise ValueError("EEGArtifactDataset requires at least one ProcessingContext")
        if target_type not in {"clean", "artifact"}:
            raise ValueError(f"target_type must be 'clean' or 'artifact', got '{target_type}'")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")

        self.contexts = list(contexts)
        self.chunk_size = int(chunk_size)
        self.target_type = target_type
        self.trigger_aligned = trigger_aligned
        self.overlap = float(overlap)
        self.transforms: list[Callable] = list(transforms or [])
        self.eeg_only = eeg_only

        # Chunk index: list of (noisy_array, target_array) — loaded eagerly
        self._chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        for ctx in self.contexts:
            self._extract_from_context(ctx)

    def _extract_from_context(self, ctx: ProcessingContext) -> None:
        raw = ctx.get_raw()
        raw_orig = ctx.get_raw_original()
        triggers = ctx.get_triggers()

        picks = (
            mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
            if self.eeg_only
            else np.arange(len(raw.ch_names))
        )
        if len(picks) == 0:
            return

        noisy_data = raw._data[picks].astype(np.float32)
        clean_data = raw_orig._data[picks].astype(np.float32)
        n_samples = noisy_data.shape[1]

        if self.trigger_aligned and triggers is not None and len(triggers) > 0:
            starts = self._trigger_starts(triggers, n_samples)
        else:
            starts = self._sliding_starts(n_samples)

        for start in starts:
            end = start + self.chunk_size
            if end > n_samples:
                break
            noisy_chunk = noisy_data[:, start:end]
            clean_chunk = clean_data[:, start:end]
            target_chunk = (
                noisy_chunk - clean_chunk
                if self.target_type == "artifact"
                else clean_chunk
            )
            self._chunks.append((noisy_chunk.copy(), target_chunk.copy()))

    def _trigger_starts(self, triggers: np.ndarray, n_samples: int) -> list[int]:
        return [
            int(t)
            for t in np.sort(triggers)
            if int(t) + self.chunk_size <= n_samples
        ]

    def _sliding_starts(self, n_samples: int) -> list[int]:
        hop = max(1, int(self.chunk_size * (1.0 - self.overlap)))
        return list(range(0, n_samples - self.chunk_size + 1, hop))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        noisy, target = self._chunks[idx]
        # Copy to avoid mutating the cached arrays
        noisy, target = noisy.copy(), target.copy()
        for transform in self.transforms:
            noisy, target = transform(noisy, target)
        return noisy, target

    @property
    def n_channels(self) -> int:
        """Number of EEG channels per chunk."""
        if not self._chunks:
            return 0
        return self._chunks[0][0].shape[0]

    @property
    def n_chunks(self) -> int:
        """Total number of chunks in the dataset."""
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Train / validation split
    # ------------------------------------------------------------------

    def train_val_split(
        self, val_ratio: float = 0.2, seed: int = 42
    ) -> tuple[_SubsetDataset, _SubsetDataset]:
        """Split into train and validation subsets (index-mapped, no data copy).

        Parameters
        ----------
        val_ratio : float
            Fraction of chunks reserved for validation.
        seed : int
            Random seed for the shuffle.

        Returns
        -------
        train_dataset, val_dataset
            Both are lightweight :class:`_SubsetDataset` views that share
            the underlying chunk list with this dataset.
        """
        n = len(self._chunks)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n).tolist()
        n_val = max(1, int(n * val_ratio))
        val_idx = set(indices[:n_val])
        train_idx = [i for i in range(n) if i not in val_idx]
        val_idx_list = [i for i in range(n) if i in val_idx]
        return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx_list)

    # ------------------------------------------------------------------
    # Framework adapters
    # ------------------------------------------------------------------

    def to_torch(self) -> Any:
        """Return a ``torch.utils.data.Dataset`` wrapping this dataset.

        Requires PyTorch to be installed.

        Example
        -------
        ::

            loader = DataLoader(dataset.to_torch(), batch_size=16, shuffle=True)
        """
        return _TorchDatasetAdapter(self)

    def to_tf(self, batch_size: int = 16) -> Any:
        """Return a ``tf.data.Dataset`` from this dataset.

        Requires TensorFlow to be installed.
        """
        return _build_tf_dataset(self, batch_size)

    def __repr__(self) -> str:
        return (
            f"EEGArtifactDataset("
            f"n_chunks={len(self._chunks)}, "
            f"n_channels={self.n_channels}, "
            f"chunk_size={self.chunk_size}, "
            f"target_type='{self.target_type}', "
            f"trigger_aligned={self.trigger_aligned})"
        )


# ---------------------------------------------------------------------------
# Framework adapter — PyTorch
# ---------------------------------------------------------------------------


class _TorchDatasetAdapter:
    """Thin ``torch.utils.data.Dataset`` wrapper around any duck-typed dataset."""

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for to_torch(). "
                "Install it with: pip install torch"
            ) from exc

        noisy, target = self._dataset[idx]
        return torch.as_tensor(noisy, dtype=torch.float32), torch.as_tensor(
            target, dtype=torch.float32
        )


# ---------------------------------------------------------------------------
# Framework adapter — TensorFlow
# ---------------------------------------------------------------------------


def _build_tf_dataset(dataset: Any, batch_size: int) -> Any:
    """Build a ``tf.data.Dataset`` from a duck-typed dataset."""
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for to_tf(). "
            "Install it with: pip install tensorflow"
        ) from exc

    noisy_0, _ = dataset[0]
    n_channels, chunk_size = noisy_0.shape

    def _generator():
        for i in range(len(dataset)):
            noisy, target = dataset[i]
            yield noisy.astype("float32"), target.astype("float32")

    output_signature = (
        tf.TensorSpec(shape=(n_channels, chunk_size), dtype=tf.float32),
        tf.TensorSpec(shape=(n_channels, chunk_size), dtype=tf.float32),
    )
    tf_ds = tf.data.Dataset.from_generator(_generator, output_signature=output_signature)
    return tf_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
