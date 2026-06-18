"""Build the AAS-decoupled spatio-temporal reference dataset (Run 3 / Weg A).

This is the data foundation of ``docs/research/run_3_decoupled_dataset_weg_a.md``.
It turns a *single* artifact bundle (e.g. ``niazy_aas_pca4_direct`` produced by
``tools/dataset_building/extract_niazy_aas_pca4_artifact.py``) into a
``(N, context_epochs, 3, S+2G)`` training set where every example is one
*(target channel ``c``, center epoch ``e``)* pair:

* input  = the real cross-channel reference of the target channel and its two
  nearest montage neighbours over ``context_epochs`` consecutive epochs,
* target = the artifact of ``(c, e)`` (or the independent clean of ``(c, e)``).

Two decoupling levers (run_3 §2/§3) are realised here:

1. **More complete artifact** — supplied upstream: the bundle's ``artifact`` is
   already ``AAS + PCA/OBS(n_components=4, hp_freq=300)``, not AAS alone.
2. **Independent clean** — ``clean_source`` selects where the clean target comes
   from: ``synthetic`` (default for decoupling), ``external`` (real clean of the
   same montage) or ``aas_corrected`` (the old, coupled baseline).

The arrays are stored with a **guard band** of ``guard_samples`` resampled
samples on each side so the online :class:`~facet.training.dataset.WindowShift`
transform can re-crop the window from the (locally) continuous signal instead of
circularly rolling it. See run_3 §5 (Kern-Shift).

No 70 Hz low-pass is applied anywhere in this path (run_3 §2): the artifact is
learned in its entirety, including the >300 Hz OBS residual.
"""

from __future__ import annotations

import logging
from math import gcd
from typing import Any

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

# Niazy uses old 10-20 names; map to modern 10-05 montage equivalents.
# Kept in sync with facet.models.st_gnn_paper_accurate_edition.training.
_LEGACY_NAME_ALIAS: dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

CLEAN_SOURCES = ("synthetic", "external", "aas_corrected")


# ---------------------------------------------------------------------------
# Low-level signal helpers
# ---------------------------------------------------------------------------


def _resample_1d(values: np.ndarray, target_samples: int) -> np.ndarray:
    """Band-limited polyphase resampling (matches the proof-fit builder).

    Uses ``scipy.signal.resample_poly`` so training data and inference share the
    same canonical resampler and HF artifact content is preserved.
    """
    n = int(values.shape[-1])
    if n == target_samples:
        return values.astype(np.float32, copy=True)
    if n == 0:
        return np.zeros(target_samples, dtype=np.float32)
    g = gcd(target_samples, n)
    up, down = target_samples // g, n // g
    out = resample_poly(values.astype(np.float64, copy=False), up, down)
    if out.shape[-1] > target_samples:
        out = out[..., :target_samples]
    elif out.shape[-1] < target_samples:
        pad = target_samples - out.shape[-1]
        out = np.concatenate([out, np.full(pad, out[..., -1], dtype=out.dtype)], axis=-1)
    return out.astype(np.float32, copy=False)


def epoch_boundaries(
    triggers: np.ndarray,
    sfreq: float,
    offset_seconds: float,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trigger-to-trigger epoch ``(starts, stops)`` clipped to the signal.

    Identical convention to ``build_niazy_proof_fit_context_dataset`` so the two
    datasets share the same epoch grid.
    """
    triggers = np.asarray(triggers, dtype=np.int64)
    if triggers.size < 2:
        raise ValueError("At least two triggers are required")
    offset_samples = int(round(offset_seconds * sfreq))
    starts = triggers[:-1] + offset_samples
    stops = triggers[1:] + offset_samples
    valid = (starts >= 0) & (stops > starts) & (stops <= n_samples)
    starts = starts[valid].astype(np.int64)
    stops = stops[valid].astype(np.int64)
    if starts.size == 0:
        raise ValueError("No valid trigger-to-trigger epochs remained after clipping")
    return starts, stops


def _guard_window(
    signal: np.ndarray,
    start: int,
    stop: int,
    guard_native: int,
    out_len: int,
) -> np.ndarray:
    """Resample the native epoch *plus a guard band* to ``out_len`` samples.

    The native window ``[start - guard_native : stop + guard_native]`` is
    edge-padded where it runs past the recording, then resampled to ``out_len``
    (``= core_samples + 2 * guard_samples``). Because the resampling ratio is
    preserved, integer-cropping the result by ``δ`` samples is exactly the same
    as cutting the native window ``δ`` resampled-samples earlier/later — the
    correct continuous-window shift run_3 §5 asks for.
    """
    n = int(signal.shape[-1])
    a, b = start - guard_native, stop + guard_native
    left_pad = max(0, -a)
    right_pad = max(0, b - n)
    seg = signal[max(0, a):min(n, b)]
    if left_pad or right_pad:
        seg = np.pad(seg, (left_pad, right_pad), mode="edge")
    return _resample_1d(seg, out_len)


# ---------------------------------------------------------------------------
# Montage neighbours (k-NN), reused from st_gnn's geodesic adjacency
# ---------------------------------------------------------------------------


def _channel_positions(ch_names: list[str]) -> np.ndarray | None:
    """3-D electrode positions on ``standard_1005`` (``None`` if unavailable)."""
    try:
        import mne  # noqa: PLC0415

        montage = mne.channels.make_standard_montage("standard_1005")
        positions = montage.get_positions()["ch_pos"]
    except Exception as exc:  # pragma: no cover - mne always present in repo
        logger.warning("Could not load standard_1005 montage (%s); using index layout", exc)
        return None

    out = np.zeros((len(ch_names), 3), dtype=np.float64)
    for idx, name in enumerate(ch_names):
        lookup = _LEGACY_NAME_ALIAS.get(name, name)
        if lookup not in positions:
            logger.warning("Channel %r (alias %r) not in standard_1005; using index layout", name, lookup)
            return None
        out[idx] = positions[lookup]
    return out


def select_neighbors(ch_names: list[str], k_neighbors: int = 2) -> np.ndarray:
    """``(n_channels, 1 + k_neighbors)`` indices ``[c, n1, ..., nk]`` per channel.

    Neighbours are the ``k`` nearest electrodes by geodesic (great-circle)
    distance on the unit sphere — the same spatial notion ``st_gnn`` uses. When
    montage positions are unavailable, falls back to nearest channel *indices*
    (a 1-D line layout) and logs a warning.
    """
    n = len(ch_names)
    if n < 1 + k_neighbors:
        raise ValueError(f"Need at least {1 + k_neighbors} channels for k_neighbors={k_neighbors}, got {n}")

    positions = _channel_positions(ch_names)
    if positions is not None:
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        unit = positions / norms
        cosine = np.clip(unit @ unit.T, -1.0, 1.0)
        dist = np.arccos(cosine)  # geodesic distance in radians
    else:
        idx = np.arange(n, dtype=np.float64)
        dist = np.abs(idx[:, None] - idx[None, :])

    np.fill_diagonal(dist, 0.0)
    out = np.zeros((n, 1 + k_neighbors), dtype=np.int64)
    for c in range(n):
        order = np.argsort(dist[c], kind="stable")  # self (dist 0) first
        out[c] = order[: 1 + k_neighbors]
    return out


# ---------------------------------------------------------------------------
# Independent clean source + spike injection
# ---------------------------------------------------------------------------


def synthetic_clean(
    n_channels: int,
    n_samples: int,
    sfreq: float,
    *,
    target_rms: float,
    seed: int = 0,
) -> np.ndarray:
    """Cheap but spectrally plausible synthetic clean EEG ``(n_channels, n_samples)``.

    1/f-ish background (cumulative white noise, drift-removed) plus a ~10 Hz alpha
    bump, scaled per channel to ``target_rms``. Deliberately self-contained so the
    builder has no dependency on the example spike/EEG generators.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / float(sfreq)
    out = np.empty((n_channels, n_samples), dtype=np.float64)
    for ch in range(n_channels):
        pink = np.cumsum(rng.standard_normal(n_samples))
        pink -= np.linspace(pink[0], pink[-1], n_samples)  # remove the random-walk drift
        alpha = np.sin(2.0 * np.pi * (8.0 + 4.0 * rng.random()) * t + 2.0 * np.pi * rng.random())
        sig = pink + 0.5 * np.std(pink) * alpha
        rms = float(np.sqrt(np.mean(sig**2))) or 1.0
        out[ch] = sig * (target_rms / rms)
    return out.astype(np.float32)


def inject_spikes(
    clean: np.ndarray,
    sfreq: float,
    *,
    rate_hz: float,
    amplitude: float,
    width_ms: float,
    seed: int = 0,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Add parametric epileptiform-like spikes to ``clean`` in place-safe copy.

    Returns the spiked signal and a list of ``(channel, sample)`` spike centers
    so the builder can derive the ``spike_labels`` mask. Spikes are sharp
    Gaussian-derivative bumps — the known events run_6 measures preservation on.
    """
    rng = np.random.default_rng(seed)
    out = clean.astype(np.float32).copy()
    n_channels, n_samples = out.shape
    half = max(1, int(round(0.5 * width_ms * 1e-3 * sfreq)))
    x = np.arange(-half, half + 1)
    kernel = (-x / max(half, 1)) * np.exp(-(x**2) / (2.0 * (half / 2.0) ** 2))
    kernel = kernel.astype(np.float32)
    centers: list[tuple[int, int]] = []
    duration_s = n_samples / float(sfreq)
    for ch in range(n_channels):
        n_spikes = rng.poisson(rate_hz * duration_s)
        for _ in range(int(n_spikes)):
            center = int(rng.integers(half, n_samples - half))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            scale = amplitude * (0.6 + 0.8 * rng.random())
            out[ch, center - half: center + half + 1] += sign * scale * kernel
            centers.append((ch, center))
    return out, centers


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_spatiotemporal_reference_dataset(
    bundle: dict[str, np.ndarray],
    *,
    context_epochs: int = 7,
    core_samples: int = 512,
    guard_samples: int = 16,
    k_neighbors: int = 2,
    clean_source: str = "synthetic",
    external_clean: np.ndarray | None = None,
    inject_spikes_mode: bool = False,
    spike_rate_hz: float = 0.7,
    spike_amplitude_uv: float = 40.0,
    spike_width_ms: float = 20.0,
    spike_label_halfwidth: int = 3,
    max_examples: int | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build the ``(N, context_epochs, 1+k, S+2G)`` decoupled reference dataset.

    Parameters mirror run_3 §4-§6. ``clean_source`` chooses the independent clean
    (``synthetic`` recommended for decoupling); ``inject_spikes_mode`` activates
    the run_6 spike-preservation foundation (§6.6) and only applies when the clean
    is synthetic.

    Returns a dict of arrays ready for ``np.savez_compressed`` and consumption by
    :class:`~facet.training.dataset.NPZSpatioTemporalDataset`.
    """
    if context_epochs < 3 or context_epochs % 2 == 0:
        raise ValueError("context_epochs must be an odd integer >= 3")
    if core_samples < 8:
        raise ValueError("core_samples must be at least 8")
    if guard_samples < 0:
        raise ValueError("guard_samples must be >= 0")
    if clean_source not in CLEAN_SOURCES:
        raise ValueError(f"clean_source must be one of {CLEAN_SOURCES}, got {clean_source!r}")

    artifact = np.asarray(bundle["artifact"], dtype=np.float32)
    corrected = np.asarray(bundle["corrected"], dtype=np.float32)
    if artifact.shape != corrected.shape:
        raise ValueError(f"artifact {artifact.shape} and corrected {corrected.shape} must match")
    n_channels, n_samples = artifact.shape
    sfreq = float(np.asarray(bundle["sfreq"]).ravel()[0])
    offset_seconds = float(np.asarray(bundle["artifact_to_trigger_offset"]).ravel()[0])
    ch_names = [str(c) for c in np.asarray(bundle["ch_names"]).tolist()]

    spikes_enabled = bool(inject_spikes_mode) and clean_source == "synthetic"
    if inject_spikes_mode and not spikes_enabled:
        logger.warning("inject_spikes_mode ignored: it requires clean_source='synthetic'")

    # --- independent clean source (run_3 §3) ---
    spike_centers: list[tuple[int, int]] = []
    if clean_source == "aas_corrected":
        clean_true = corrected
    elif clean_source == "external":
        if external_clean is None:
            raise ValueError("clean_source='external' requires external_clean of shape (n_channels, n_samples)")
        clean_true = np.asarray(external_clean, dtype=np.float32)
        if clean_true.shape != artifact.shape:
            raise ValueError(f"external_clean {clean_true.shape} must match artifact {artifact.shape}")
    else:  # synthetic
        target_rms = float(np.sqrt(np.mean(corrected.astype(np.float64) ** 2))) or 1.0
        clean_true = synthetic_clean(n_channels, n_samples, sfreq, target_rms=target_rms, seed=seed)
        if spikes_enabled:
            clean_true, spike_centers = inject_spikes(
                clean_true,
                sfreq,
                rate_hz=spike_rate_hz,
                amplitude=spike_amplitude_uv * 1e-6,
                width_ms=spike_width_ms,
                seed=seed + 1,
            )

    spikes_by_channel: dict[int, np.ndarray] = {}
    if spike_centers:
        for ch, sample in spike_centers:
            spikes_by_channel.setdefault(ch, []).append(sample)
        spikes_by_channel = {ch: np.asarray(v, dtype=np.int64) for ch, v in spikes_by_channel.items()}

    # --- epoch grid + guard geometry ---
    starts, stops = epoch_boundaries(bundle["triggers"], sfreq, offset_seconds, n_samples)
    n_epochs = starts.size
    radius = context_epochs // 2
    if n_epochs <= 2 * radius:
        raise ValueError(f"Not enough epochs ({n_epochs}) for context width {context_epochs}")
    out_len = core_samples + 2 * guard_samples
    neighbors = select_neighbors(ch_names, k_neighbors=k_neighbors)
    eeg_channels = list(range(n_channels))

    clean_ex: list[np.ndarray] = []
    artifact_ex: list[np.ndarray] = []
    target_ex: list[np.ndarray] = []
    spike_ex: list[np.ndarray] = []
    neigh_ex: list[np.ndarray] = []
    tgt_ch_ex: list[int] = []
    center_ep_ex: list[int] = []

    def _ctx_block(source: np.ndarray, chans: np.ndarray, center_idx: int) -> np.ndarray:
        block = np.empty((context_epochs, chans.size, out_len), dtype=np.float32)
        for ei, ep in enumerate(range(center_idx - radius, center_idx + radius + 1)):
            s, e = int(starts[ep]), int(stops[ep])
            g_native = int(round(guard_samples * (e - s) / core_samples))
            for ci, ch in enumerate(chans):
                block[ei, ci] = _guard_window(source[int(ch)], s, e, g_native, out_len)
        return block

    def _spike_mask(ch: int, center_idx: int) -> np.ndarray:
        mask = np.zeros((1, out_len), dtype=np.float32)
        positions = spikes_by_channel.get(int(ch))
        if positions is None:
            return mask
        s, e = int(starts[center_idx]), int(stops[center_idx])
        g_native = int(round(guard_samples * (e - s) / core_samples))
        a, b = s - g_native, e + g_native
        span = max(1, b - a)
        in_win = positions[(positions >= a) & (positions < b)]
        for p in in_win:
            idx = int(round((p - a) * out_len / span))
            lo = max(0, idx - spike_label_halfwidth)
            hi = min(out_len, idx + spike_label_halfwidth + 1)
            mask[0, lo:hi] = 1.0
        return mask

    # Center-epoch outer, channel inner: a max_examples cap then yields a
    # representative cross-section of *all* channels over the first epochs,
    # rather than every epoch of only the first few channels.
    stop = False
    for center_idx in range(radius, n_epochs - radius):
        for c in eeg_channels:
            chans = neighbors[c]  # [c, n1, ..., nk]
            clean_block = _ctx_block(clean_true, chans, center_idx)
            artifact_block = _ctx_block(artifact, chans, center_idx)

            clean_ex.append(clean_block)
            artifact_ex.append(artifact_block)
            target_ex.append(artifact_block[radius, 0:1].copy())  # artifact of (c, center epoch)
            spike_ex.append(_spike_mask(c, center_idx) if spikes_enabled else np.zeros((1, out_len), np.float32))
            neigh_ex.append(chans.copy())
            tgt_ch_ex.append(int(c))
            center_ep_ex.append(int(center_idx))

            if max_examples is not None and len(clean_ex) >= max_examples:
                stop = True
                break
        if stop:
            break

    # noisy is derivable (clean + artifact) and recomputed post-augmentation, so
    # it is not stored — saves a third of the on-disk size.
    clean_context = np.stack(clean_ex, axis=0)
    artifact_context = np.stack(artifact_ex, axis=0)
    artifact_center = np.stack(target_ex, axis=0)
    clean_center = clean_context[:, radius, 0:1].copy()
    spike_labels = np.stack(spike_ex, axis=0)
    n_examples = clean_context.shape[0]

    return {
        "clean_context": clean_context.astype(np.float32),
        "artifact_context": artifact_context.astype(np.float32),
        "clean_center": clean_center.astype(np.float32),
        "artifact_center": artifact_center.astype(np.float32),
        "spike_labels": spike_labels.astype(np.float32),
        "neighbor_channel_indices": np.stack(neigh_ex, axis=0).astype(np.int64),
        "target_channel_index": np.asarray(tgt_ch_ex, dtype=np.int64),
        "center_epoch_index": np.asarray(center_ep_ex, dtype=np.int64),
        "core_samples": np.asarray([core_samples], dtype=np.int64),
        "guard_samples": np.asarray([guard_samples], dtype=np.int64),
        "context_epochs": np.asarray([context_epochs], dtype=np.int64),
        "k_neighbors": np.asarray([k_neighbors], dtype=np.int64),
        "sfreq": np.asarray([sfreq], dtype=np.float64),
        "ch_names": np.asarray(ch_names, dtype=object),
        "clean_source": np.asarray([clean_source], dtype=object),
        "spikes_injected": np.asarray([spikes_enabled], dtype=bool),
        "n_examples": np.asarray([n_examples], dtype=np.int64),
    }


def summarize(dataset: dict[str, Any]) -> dict[str, Any]:
    """Compact human-readable summary for the builder CLI / metadata JSON."""
    art = dataset["artifact_context"]
    clean = dataset["clean_context"]
    noisy = clean + art
    return {
        "n_examples": int(dataset["n_examples"][0]),
        "input_shape": list(noisy.shape[1:]),
        "target_shape": list(dataset["artifact_center"].shape[1:]),
        "context_epochs": int(dataset["context_epochs"][0]),
        "core_samples": int(dataset["core_samples"][0]),
        "guard_samples": int(dataset["guard_samples"][0]),
        "k_neighbors": int(dataset["k_neighbors"][0]),
        "clean_source": str(dataset["clean_source"][0]),
        "spikes_injected": bool(dataset["spikes_injected"][0]),
        "spike_label_positive_fraction": float(np.mean(dataset["spike_labels"] > 0)),
        "sampling_frequency_hz": float(dataset["sfreq"][0]),
        "mean_abs_clean_uv": float(np.mean(np.abs(clean)) * 1e6),
        "mean_abs_artifact_uv": float(np.mean(np.abs(art)) * 1e6),
        "mean_abs_noisy_uv": float(np.mean(np.abs(noisy)) * 1e6),
    }
