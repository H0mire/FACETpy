"""Framework-agnostic EEG artifact dataset built from FACETpy ProcessingContexts."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mne
import numpy as np

from ..core import ProcessingContext

logger = logging.getLogger(__name__)

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

    def __call__(self, noisy: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.1), seed: int = 0) -> None:
        self.scale_range = scale_range
        self._rng = np.random.default_rng(seed)

    def __call__(self, noisy: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def __call__(self, noisy: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def __call__(self, noisy: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.p <= 0.0:
            return noisy, target
        if self._rng.random() < self.p:
            return -noisy, -target
        return noisy, target


# ---------------------------------------------------------------------------
# Spatio-temporal transforms (Run 3 / Weg A)
#
# These operate on a *sample dict* with keys ``noisy``/``clean``/``artifact``
# (each ``(epochs, channels, L)``), ``target`` and ``spike`` (each ``(1, L)``),
# where ``L = core_samples + 2 * guard_samples``.  They are pure given their
# seeded RNG and return a new dict.  The dataset applies them in order, with
# :class:`WindowShift` always last (it removes the guard band, ``L -> core``).
# ---------------------------------------------------------------------------


class WindowShift:
    """Continuous-window shift: crop the guard-padded window to ``core_samples``.

    Picks an offset ``δ`` in ``[-max_shift, +max_shift]`` (capped by the guard
    band) and crops ``[guard + δ : guard + δ + core]`` from every array.  Because
    the builder stored each epoch resampled *with* a guard band, this is exactly
    cutting the window ``δ`` samples earlier/later in the continuous signal —
    the correct sub-sample shift of run_3 §5, **not** the circular ``np.roll`` of
    :class:`TriggerJitter`.

    With ``fractional=True`` a sub-sample shift is realised by polyphase
    resampling before the integer crop.
    """

    def __init__(
        self,
        core_samples: int,
        max_shift: int | None = None,
        fractional: bool = False,
        seed: int = 0,
    ) -> None:
        self.core_samples = int(core_samples)
        self.max_shift = max_shift
        self.fractional = fractional
        self._rng = np.random.default_rng(seed)

    def _pick_shift(self, guard: int) -> float:
        cap = guard if self.max_shift is None else min(int(self.max_shift), guard)
        if cap <= 0:
            return 0.0
        if self.fractional:
            return float(self._rng.uniform(-cap, cap))
        return float(self._rng.integers(-cap, cap + 1))

    @staticmethod
    def _crop(arr: np.ndarray, start: int, length: int) -> np.ndarray:
        return arr[..., start:start + length]

    @staticmethod
    def _gather(arr: np.ndarray, start_float: float, length: int) -> np.ndarray:
        """Linear-interpolated crop of ``length`` samples starting at ``start_float``."""
        positions = start_float + np.arange(length)
        max_i = arr.shape[-1] - 1
        i0 = np.clip(np.floor(positions).astype(np.int64), 0, max_i - 1)
        w = (positions - i0).astype(np.float32)
        return (arr[..., i0] * (1.0 - w) + arr[..., i0 + 1] * w).astype(np.float32)

    def __call__(self, sample: dict) -> dict:
        core = self.core_samples
        length = sample["noisy"].shape[-1]
        guard = (length - core) // 2
        if guard < 0:
            raise ValueError(f"WindowShift: window length {length} < core_samples {core}")
        delta = self._pick_shift(guard)
        keys = [k for k in ("noisy", "clean", "artifact", "target", "spike") if k in sample]
        out = dict(sample)

        if self.fractional and float(delta) != round(delta):
            start_float = float(np.clip(guard + delta, 0.0, length - core))
            for k in keys:
                out[k] = self._gather(sample[k], start_float, core)
            return out

        start = int(np.clip(guard + int(round(delta)), 0, length - core))
        for k in keys:
            out[k] = self._crop(sample[k], start, core).astype(np.float32, copy=True)
        return out


class BackgroundMix:
    """Clean-swap background mix (run_3 §5 item 1).

    With probability ``prob`` replace the clean background block with an
    *alternate* clean block from a different epoch (``e' != e``) — supplied by the
    dataset as ``sample['clean_alt']`` — and recompute ``noisy = clean_alt +
    artifact``.  Only the **whole-block epoch pairing** is randomised; channel
    identities and the per-channel spatial signature are never touched, so the
    cross-channel features the model must learn stay realistic.  The artifact and
    target are unchanged.  No-op when ``clean_alt`` is absent.
    """

    def __init__(self, prob: float = 0.5, seed: int = 0) -> None:
        self.prob = float(prob)
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict) -> dict:
        clean_alt = sample.get("clean_alt")
        if clean_alt is None or self.prob <= 0.0 or self._rng.random() >= self.prob:
            return sample
        out = dict(sample)
        out["clean"] = clean_alt.astype(np.float32, copy=True)
        out["noisy"] = (clean_alt + sample["artifact"]).astype(np.float32)
        return out


class AmplitudeJitter:
    """Global *and* per-channel amplitude jitter (run_3 §5 item 2).

    Scales ``noisy``/``clean``/``artifact`` and the ``target`` by a global scalar
    times per-channel gains (gradient-gain drift).  Applied consistently so
    ``noisy = clean + artifact`` and ``target = artifact[center, target_ch]`` stay
    intact.  The spike mask is left untouched.
    """

    def __init__(
        self,
        global_range: tuple[float, float] = (0.9, 1.1),
        per_channel_range: tuple[float, float] = (0.95, 1.05),
        seed: int = 0,
    ) -> None:
        self.global_range = global_range
        self.per_channel_range = per_channel_range
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict) -> dict:
        noisy = sample["noisy"]
        n_channels = noisy.shape[1]
        g = float(self._rng.uniform(*self.global_range))
        gains = self._rng.uniform(*self.per_channel_range, size=n_channels).astype(np.float32)
        ch_scale = (g * gains)[None, :, None]  # (1, C, 1) broadcast over (E, C, L)

        out = dict(sample)
        for k in ("noisy", "clean", "artifact"):
            if k in sample:
                out[k] = (sample[k] * ch_scale).astype(np.float32)
        if "target" in sample:  # target channel is index 0 of the triple
            out["target"] = (sample["target"] * (g * gains[0])).astype(np.float32)
        return out


class LengthJitterNoise:
    """TR/length jitter + measurement noise (run_3 §5 item 4).

    Time-warps the window by ``1 ± length_eps`` (resample then crop/pad back to
    the original length) consistently across all signals and the spike mask, then
    adds small Gaussian measurement noise and a linear baseline drift to the
    **noisy** signal only.  The noise is deliberately absent from clean/artifact/
    target: it is real acquisition noise the artifact-predictor should *not*
    reproduce.
    """

    def __init__(
        self,
        length_eps: float = 0.02,
        noise_std_frac: float = 0.01,
        baseline_drift_frac: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.length_eps = float(length_eps)
        self.noise_std_frac = float(noise_std_frac)
        self.baseline_drift_frac = float(baseline_drift_frac)
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _resample_to(arr: np.ndarray, target_len: int) -> np.ndarray:
        from math import gcd  # noqa: PLC0415

        from scipy.signal import resample_poly  # noqa: PLC0415

        n = arr.shape[-1]
        if n == target_len:
            return arr
        g = gcd(target_len, n)
        return resample_poly(arr, target_len // g, n // g, axis=-1).astype(np.float32)

    def __call__(self, sample: dict) -> dict:
        out = dict(sample)
        length = sample["noisy"].shape[-1]

        if self.length_eps > 0:
            eps = float(self._rng.uniform(-self.length_eps, self.length_eps))
            warped = max(8, int(round(length * (1.0 + eps))))
            for k in ("noisy", "clean", "artifact", "target", "spike"):
                if k not in sample:
                    continue
                w = self._resample_to(sample[k], warped)
                if warped >= length:
                    out[k] = w[..., :length]
                else:
                    pad = [(0, 0)] * w.ndim
                    pad[-1] = (0, length - warped)
                    out[k] = np.pad(w, pad, mode="edge")
            out["spike"] = (out["spike"] > 0.5).astype(np.float32) if "spike" in out else out.get("spike")

        noisy = out["noisy"]
        if self.noise_std_frac > 0:
            std = float(np.std(noisy)) * self.noise_std_frac
            noisy = noisy + self._rng.normal(0.0, std, size=noisy.shape).astype(np.float32)
        if self.baseline_drift_frac > 0:
            amp = float(np.std(noisy)) * self.baseline_drift_frac
            ramp = np.linspace(-amp, amp, noisy.shape[-1], dtype=np.float32)
            noisy = noisy + ramp
        out["noisy"] = noisy.astype(np.float32)
        return out


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
    ``examples/dataset_building/build_synthetic_spike_artifact_context_dataset.py``. Each item
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
        target_extras: tuple[str, ...] = (),
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        # facet-train hands the loss only ``(prediction, target)``, so an objective
        # that scores the *recovered clean* -- see
        # :mod:`facet.training.deployment_losses` -- can only reach the clean and
        # noisy signals if they ride along as extra target rows. Default is the
        # empty tuple, so every existing config keeps its exact contract.
        _EXTRA_KEYS = {"clean": "clean_center", "noisy": "noisy_center",
                       "artifact": "artifact_center"}
        for extra in target_extras:
            if extra not in _EXTRA_KEYS:
                raise ValueError(
                    f"target_extras entries must be one of {sorted(_EXTRA_KEYS)}, got {extra!r}")
        self.target_extras = tuple(target_extras)

        with np.load(self.path, allow_pickle=False) as bundle:
            self.noisy = bundle[input_key].astype(np.float32, copy=False)
            self.target = bundle[target_key].astype(np.float32, copy=False)
            self.extras = {name: bundle[_EXTRA_KEYS[name]].astype(np.float32, copy=False)
                           for name in self.target_extras}
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
            self.extras = {k: v[:limit] for k, v in self.extras.items()}

        self.context_epochs = int(self.noisy.shape[1])
        self.n_channels = int(self.noisy.shape[2])
        self.epoch_samples = int(self.noisy.shape[3])
        self.chunk_size = self.epoch_samples
        self.target_key = target_key
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
        if self.target_extras:
            rows = [target]
            for name in self.target_extras:
                row = self.extras[name][idx].copy()
                # The extras have to live in the same space as the target, or
                # ``noisy - prediction`` is off by whatever mean was removed from
                # one and not the other -- a silent constant offset, which is the
                # exact defect the seam-step metric was written to catch.
                if self.demean_target:
                    row -= row.mean(axis=-1, keepdims=True)
                rows.append(row)
            target = np.stack(rows, axis=0)
        return noisy, target

    @property
    def n_chunks(self) -> int:
        return len(self)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.context_epochs, self.n_channels, self.epoch_samples)

    @property
    def target_rows(self) -> tuple[str, ...]:
        """Names of the stacked target rows, in order, for the loss to index by.

        The primary row is named after ``target_key`` so a loss configured with
        ``rows=dataset.target_rows`` cannot disagree with the dataset about which
        row is which.
        """
        primary = {"artifact_center": "artifact", "clean_center": "clean",
                   "noisy_center": "noisy"}.get(self.target_key, self.target_type)
        return (primary, *self.target_extras)

    @property
    def target_shape(self) -> tuple[int, ...]:
        if self.target_extras:
            return (1 + len(self.target_extras), self.n_channels, self.epoch_samples)
        return (self.n_channels, self.epoch_samples)

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42) -> tuple[_SubsetDataset, _SubsetDataset]:
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


def _proportional_prefix(split: np.ndarray | None, n_total: int, limit: int) -> np.ndarray:
    """Indices of the first ``limit`` examples, spread over the stored splits.

    Without a stored split this is simply ``arange(limit)``. With one, each split
    keeps its own share of *limit* — so a subsampled run still has a validation set
    — and each share is a prefix of that split, so the contiguous ordering the
    guard band relies on survives.
    """
    limit = max(0, min(int(limit), int(n_total)))
    if split is None:
        return np.arange(limit, dtype=np.int64)
    keep: list[np.ndarray] = []
    for value in np.unique(split):
        member = np.flatnonzero(split == value)
        share = int(round(limit * member.size / max(1, n_total)))
        keep.append(member[: max(1, share) if limit > 0 else 0])
    return np.sort(np.concatenate(keep)) if keep else np.arange(0, dtype=np.int64)


class NPZSpatioTemporalDataset:
    """AAS-decoupled spatio-temporal reference dataset (Run 3 / Weg A).

    Reads the ``.npz`` produced by
    :func:`facet.training.spatiotemporal_builder.build_spatiotemporal_reference_dataset`.
    Each example is one *(target channel, center epoch)* pair: the input is the
    guard-padded cross-channel reference ``(context_epochs, 1 + k_neighbors, L)``
    and the target is the artifact (or clean) of the target channel at the center
    epoch ``(1, L)``, with ``L = core_samples + 2 * guard_samples``.

    Online augmentation pipeline per ``__getitem__``:

    1. optional **background mix** (clean-swap, ``background_mix_prob``) — swaps
       the clean background with another epoch's clean block of the *same* channel
       triple, then recomputes ``noisy = clean_alt + artifact``;
    2. user ``transforms`` (e.g. :class:`AmplitudeJitter`, :class:`LengthJitterNoise`);
    3. a final :class:`WindowShift` that crops the guard band away (``L -> core``)
       with a random continuous-window offset of up to ``max_shift`` samples.

    The returned arrays therefore have the guard removed:
    ``(context_epochs, 1 + k_neighbors, core_samples)`` and ``(1, core_samples)``.
    """

    def __init__(
        self,
        path: str | Path,
        target_key: str = "artifact_center",
        max_examples: int | None = None,
        transforms: list[Callable] | None = None,
        max_shift: int | None = None,
        fractional_shift: bool = False,
        background_mix_prob: float = 0.0,
        demean_input: bool = False,
        demean_target: bool = False,
        target_extras: tuple[str, ...] = (),
        max_channels: int | None = None,
        residual_mode: bool = False,
        seed: int = 0,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if target_key not in {"artifact_center", "clean_center"}:
            raise ValueError(f"target_key must be 'artifact_center' or 'clean_center', got {target_key!r}")

        with np.load(self.path, allow_pickle=True) as b:
            # noisy is derivable (clean + artifact) and recomputed per item, so it
            # is neither stored on disk nor held as a third full array in RAM.
            self.clean = b["clean_context"].astype(np.float32, copy=False)
            self.artifact = b["artifact_context"].astype(np.float32, copy=False)
            self.target = b[target_key].astype(np.float32, copy=False)
            self.spike = (
                b["spike_labels"].astype(np.float32, copy=False)
                if "spike_labels" in b
                else np.zeros((self.clean.shape[0], 1, self.clean.shape[-1]), dtype=np.float32)
            )
            self.neighbor_channel_indices = b["neighbor_channel_indices"].astype(np.int64, copy=False)
            self.center_epoch_index = b["center_epoch_index"].astype(np.int64, copy=False)
            # Builder-supplied leakage-free split (0=train, 1=val); absent in
            # datasets built before run_3 §7.
            self.example_split = (
                b["example_split"].astype(np.int64, copy=False) if "example_split" in b else None
            )
            # AAS/FARM-removable part of the artifact, per context epoch and centre
            # epoch; present in datasets built after run_6 §3c.
            self.template_ctx = (
                b["artifact_context_template"].astype(np.float32, copy=False)
                if "artifact_context_template" in b
                else None
            )
            self.template_center = (
                b["artifact_center_template"].astype(np.float32, copy=False)
                if "artifact_center_template" in b
                else None
            )
            self.core_samples = int(np.asarray(b["core_samples"]).ravel()[0])
            self.guard_samples = int(np.asarray(b["guard_samples"]).ravel()[0])
            self.context_epochs = int(np.asarray(b["context_epochs"]).ravel()[0])
            self.k_neighbors = int(np.asarray(b["k_neighbors"]).ravel()[0])
            self.sfreq = float(np.asarray(b["sfreq"]).ravel()[0])

        if self.clean.ndim != 4:
            raise ValueError("clean_context must have shape (examples, context_epochs, channels, L)")

        if max_examples is not None:
            # Take a proportional prefix from *each* stored split, not a global one.
            # The builder writes the validation examples as a contiguous tail, so a
            # plain [:limit] on a subsampled run keeps only training examples and the
            # validation set comes out empty — which surfaced as an unpack error deep
            # inside the trainer rather than as anything about the data. Prefixes
            # within each split preserve the guard band and the leakage-free boundary.
            keep = _proportional_prefix(self.example_split, self.clean.shape[0], int(max_examples))
            self.clean, self.artifact = self.clean[keep], self.artifact[keep]
            self.target, self.spike = self.target[keep], self.spike[keep]
            self.neighbor_channel_indices = self.neighbor_channel_indices[keep]
            self.center_epoch_index = self.center_epoch_index[keep]
            if self.example_split is not None:
                self.example_split = self.example_split[keep]
            if self.template_ctx is not None:
                self.template_ctx = self.template_ctx[keep]
            if self.template_center is not None:
                self.template_center = self.template_center[keep]

        if max_channels is not None:
            # Keep only the first ``max_channels`` electrodes of the context. The
            # builder writes the target channel first and its geodesic neighbours
            # after it, so a prefix is "target plus the nearest N-1 neighbours" and
            # ``max_channels=1`` is the target channel alone. This is the knob for
            # ablating spatial context, and for running a model in the
            # single-channel configuration its source paper actually used.
            limit = max(1, min(int(max_channels), self.clean.shape[2]))
            self.clean = self.clean[:, :, :limit]
            self.artifact = self.artifact[:, :, :limit]
            self.neighbor_channel_indices = self.neighbor_channel_indices[:, :limit]
            if self.template_ctx is not None:
                self.template_ctx = self.template_ctx[:, :, :limit]

        self.target_key = target_key
        # facet-train hands the loss only (prediction, target), so anything a loss
        # needs beyond the target itself has to ride along as extra target rows.
        # Order is always [target, *target_extras]; the (window-shifted) clean and
        # spike mask stay aligned with the target because WindowShift crops them
        # together. ``("clean", "spike")`` is what the recovered-clean objective
        # needs: it reconstructs noisy = clean + artifact and scores what the model
        # would actually hand back to a clinician.
        for extra in target_extras:
            if extra not in {"clean", "spike"}:
                raise ValueError(f"target_extras entries must be 'clean' or 'spike', got {extra!r}")
        self.target_extras = tuple(target_extras)
        # Cascade formulation (run_6 §3c). Trying to predict the whole artifact is a
        # ~40 dB separation problem — 1961 µV of artifact over 19 µV of EEG — and the
        # models never solved it: correlation with the true clean stayed near zero
        # whatever the loss. FARM does not have that problem because its estimate is
        # an average over observed epochs, so a non-periodic IED cannot enter it and
        # the EEG survives by construction.
        #
        # In residual mode the model instead sees the FARM-corrected signal and
        # predicts only what FARM leaves behind: the artifact drops to ~111 µV, the
        # ratio to the EEG from ~100:1 to ~6:1, and "beat FARM" becomes the literal
        # objective rather than "separate from scratch".
        self.residual_mode = bool(residual_mode)
        if self.residual_mode and (self.template_ctx is None or self.template_center is None):
            raise ValueError(
                "residual_mode requires artifact_context_template and artifact_center_template; "
                "rebuild the dataset with the current builder"
            )
        self.target_type = "artifact" if target_key == "artifact_center" else "clean"
        self.chunk_size = self.core_samples
        self.trigger_aligned = True
        self.demean_input = demean_input
        self.demean_target = demean_target
        self.transforms: list[Callable] = list(transforms or [])

        self.background_mix_prob = float(background_mix_prob)
        if self.background_mix_prob > 0.0 and self.target_key == "clean_center":
            logger.warning("background_mix disabled: it is incompatible with target_key='clean_center'")
            self.background_mix_prob = 0.0
        self._bg_mix = BackgroundMix(prob=self.background_mix_prob, seed=seed + 1) if self.background_mix_prob > 0 else None
        self._bg_rng = np.random.default_rng(seed + 2)
        self._alt_groups = self._build_alt_groups() if self._bg_mix is not None else {}
        self._window_shift = WindowShift(
            core_samples=self.core_samples, max_shift=max_shift, fractional=fractional_shift, seed=seed
        )

    def _build_alt_groups(self) -> dict[bytes, np.ndarray]:
        """Map channel-triple -> example indices, for same-montage clean swaps."""
        groups: dict[bytes, list[int]] = {}
        for i, triple in enumerate(self.neighbor_channel_indices):
            groups.setdefault(triple.tobytes(), []).append(i)
        return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}

    def __len__(self) -> int:
        return int(self.clean.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        clean_i, artifact_i = self.clean[idx], self.artifact[idx]
        if self.residual_mode:
            # noisy becomes clean + (artifact - template); the target becomes the
            # residual alone. Subtracting before the transforms keeps WindowShift and
            # BackgroundMix operating on exactly the signals the model will see.
            artifact_i = artifact_i - self.template_ctx[idx]
        sample = {
            "noisy": (clean_i + artifact_i).astype(np.float32),
            "clean": clean_i.copy(),
            "artifact": self.artifact[idx].copy(),
            "target": (
                (self.target[idx] - self.template_center[idx]).astype(np.float32)
                if self.residual_mode and self.target_key == "artifact_center"
                else self.target[idx].copy()
            ),
            "spike": self.spike[idx].copy(),
        }
        if self._bg_mix is not None:
            sample["clean_alt"] = self._pick_alt_clean(idx)
            sample = self._bg_mix(sample)
        for transform in self.transforms:
            sample = transform(sample)
        sample = self._window_shift(sample)

        noisy, target = sample["noisy"], sample["target"]
        if self.demean_input:
            noisy = noisy - noisy.mean(axis=-1, keepdims=True)
        if self.demean_target:
            target = target - target.mean(axis=-1, keepdims=True)
        if self.target_extras:
            rows = [target]
            for extra in self.target_extras:
                block = sample["clean"][self.context_epochs // 2, 0:1] if extra == "clean" else sample["spike"]
                rows.append(block)
            target = np.concatenate(rows, axis=0)
        return noisy.astype(np.float32), target.astype(np.float32)

    def _pick_alt_clean(self, idx: int) -> np.ndarray:
        group = self._alt_groups.get(self.neighbor_channel_indices[idx].tobytes())
        if group is None or group.size < 2:
            return self.clean[idx].copy()
        for _ in range(8):
            j = int(group[self._bg_rng.integers(group.size)])
            if j != idx:
                return self.clean[j].copy()
        return self.clean[idx].copy()

    def get_spike_labels(self, idx: int) -> np.ndarray:
        """Center-cropped spike mask ``(1, core_samples)`` for run_6 eval."""
        start = self.guard_samples
        return self.spike[idx][..., start:start + self.core_samples].copy()

    @property
    def n_channels(self) -> int:
        return int(self.clean.shape[2])

    @property
    def n_chunks(self) -> int:
        return len(self)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.context_epochs, self.n_channels, self.core_samples)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (1 + len(self.target_extras), self.core_samples)

    def train_val_split(
        self,
        val_ratio: float = 0.2,
        seed: int = 42,
        split_mode: str = "auto",
    ) -> tuple[_SubsetDataset, _SubsetDataset]:
        """Split into train and validation subsets (index-mapped, no data copy).

        Examples here are *(target channel, centre epoch)* pairs, so a naive random
        split leaks three ways at once: neighbouring centre epochs share
        ``context_epochs - 1`` of their input epochs, a tiled clean source repeats
        the same background across the recording, and a channel reappears inside its
        neighbours' inputs. Validation metrics from such a split measure
        memorisation, not generalisation (run_3 §7).

        Parameters
        ----------
        split_mode : {"auto", "stored", "contiguous", "random"}
            ``"auto"`` (default) uses the builder-supplied ``example_split`` when the
            NPZ carries one — the only mode that also accounts for the disjoint clean
            partition — and otherwise falls back to ``"contiguous"``.
            ``"contiguous"`` splits by centre epoch and drops a guard band of
            overlapping contexts at the seam. ``"random"`` is the old leaky behaviour,
            kept for ablations only and warned about.
        """
        if split_mode not in {"auto", "stored", "contiguous", "random"}:
            raise ValueError(f"split_mode must be auto/stored/contiguous/random, got {split_mode!r}")
        n = len(self)

        if split_mode in {"auto", "stored"} and self.example_split is not None:
            split = self.example_split[:n]
            train_idx = np.flatnonzero(split == 0).tolist()
            val_idx = np.flatnonzero(split == 1).tolist()
            if train_idx and val_idx:
                return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx)
            logger.warning("Stored example_split has an empty side; falling back to a contiguous split")
        elif split_mode == "stored":
            raise ValueError("split_mode='stored' requires a dataset built with example_split")

        if split_mode == "random":
            logger.warning(
                "train_val_split(split_mode='random') puts overlapping %d-epoch contexts (and, for a "
                "tiled clean, identical backgrounds) in both subsets — validation metrics will be "
                "optimistic. Use 'auto'/'contiguous' for a leakage-free split.",
                self.context_epochs,
            )
            rng = np.random.default_rng(seed)
            indices = rng.permutation(n).tolist()
            n_val = max(1, int(n * val_ratio))
            val_set = set(indices[:n_val])
            train_idx = [i for i in range(n) if i not in val_set]
            val_idx = [i for i in range(n) if i in val_set]
            return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx)

        # Contiguous split by centre epoch, dropping contexts that straddle the seam.
        epochs = self.center_epoch_index[:n]
        unique_epochs = np.unique(epochs)
        if unique_epochs.size < 2:
            raise ValueError("Need at least two distinct centre epochs for a contiguous split")
        cut_pos = int(np.clip(round(unique_epochs.size * (1.0 - val_ratio)), 1, unique_epochs.size - 1))
        cut_epoch = int(unique_epochs[cut_pos])
        guard = self.context_epochs // 2
        train_idx = np.flatnonzero(epochs + guard < cut_epoch).tolist()
        val_idx = np.flatnonzero(epochs - guard >= cut_epoch).tolist()
        n_dropped = n - len(train_idx) - len(val_idx)
        if n_dropped:
            logger.info("Contiguous split dropped %d examples in the %d-epoch guard band", n_dropped, 2 * guard)
        return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx)

    def to_torch(self) -> Any:
        return _TorchDatasetAdapter(self)

    def to_tf(self, batch_size: int = 16) -> Any:
        return _build_tf_dataset(self, batch_size)

    def __repr__(self) -> str:
        return (
            f"NPZSpatioTemporalDataset(n={len(self)}, input_shape={self.input_shape}, "
            f"target='{self.target_type}', guard={self.guard_samples}, "
            f"background_mix_prob={self.background_mix_prob})"
        )


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
            target_chunk = noisy_chunk - clean_chunk if self.target_type == "artifact" else clean_chunk
            self._chunks.append((noisy_chunk.copy(), target_chunk.copy()))

    def _trigger_starts(self, triggers: np.ndarray, n_samples: int) -> list[int]:
        return [int(t) for t in np.sort(triggers) if int(t) + self.chunk_size <= n_samples]

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
        self,
        val_ratio: float = 0.2,
        seed: int = 42,
        split_mode: str = "random",
    ) -> tuple[_SubsetDataset, _SubsetDataset]:
        """Split into train and validation subsets (index-mapped, no data copy).

        Parameters
        ----------
        val_ratio : float
            Fraction of chunks reserved for validation.
        seed : int
            Random seed for the shuffle (``split_mode="random"`` only).
        split_mode : {"random", "contiguous"}
            ``"random"`` shuffles chunks before splitting. With overlapping
            sliding windows (``overlap > 0``) this lets physically overlapping
            windows land in both subsets — temporal leakage that inflates
            validation metrics — so a warning is emitted in that case.
            ``"contiguous"`` performs a leakage-free block split (earliest
            chunks → train, latest → validation) and drops a guard band of
            overlapping windows at the seam.

        Returns
        -------
        train_dataset, val_dataset
            Both are lightweight :class:`_SubsetDataset` views that share
            the underlying chunk list with this dataset.
        """
        if split_mode not in {"random", "contiguous"}:
            raise ValueError(f"split_mode must be 'random' or 'contiguous', got {split_mode!r}")

        n = len(self._chunks)
        n_val = max(1, int(n * val_ratio))

        if split_mode == "contiguous":
            guard = self._overlap_guard_chunks()
            val_start = n - n_val
            train_idx = list(range(0, max(0, val_start - guard)))
            val_idx_list = list(range(val_start, n))
            return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx_list)

        if self.overlap > 0 and not self.trigger_aligned:
            logger.warning(
                "train_val_split(split_mode='random') with overlap=%.3f on sliding-window "
                "chunks places physically overlapping windows in both train and validation "
                "(temporal leakage -> optimistic val metrics). Pass split_mode='contiguous' "
                "for a leakage-free block split.",
                self.overlap,
            )
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n).tolist()
        val_idx = set(indices[:n_val])
        train_idx = [i for i in range(n) if i not in val_idx]
        val_idx_list = [i for i in range(n) if i in val_idx]
        return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx_list)

    def _overlap_guard_chunks(self) -> int:
        """Number of trailing train chunks to drop at the contiguous-split seam.

        Sliding windows that straddle the train/val boundary physically overlap;
        dropping ``ceil(chunk_size / hop) - 1`` chunks removes that overlap so no
        sample appears in both subsets. Returns 0 when chunks do not overlap
        (``overlap == 0``) or are trigger-aligned.
        """
        if self.overlap <= 0 or self.trigger_aligned:
            return 0
        hop = max(1, int(self.chunk_size * (1.0 - self.overlap)))
        return max(0, (self.chunk_size + hop - 1) // hop - 1)

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
            raise ImportError("PyTorch is required for to_torch(). Install it with: pip install torch") from exc

        noisy, target = self._dataset[idx]
        return torch.as_tensor(noisy, dtype=torch.float32), torch.as_tensor(target, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Framework adapter — TensorFlow
# ---------------------------------------------------------------------------


def _build_tf_dataset(dataset: Any, batch_size: int) -> Any:
    """Build a ``tf.data.Dataset`` from a duck-typed dataset."""
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("TensorFlow is required for to_tf(). Install it with: pip install tensorflow") from exc

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
