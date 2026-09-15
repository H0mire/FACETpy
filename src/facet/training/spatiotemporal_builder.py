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

CLEAN_SOURCES = ("synthetic", "external", "aas_corrected", "niazy_pretrigger")


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


def resample_and_tile(
    clean: np.ndarray,
    src_sfreq: float,
    dst_sfreq: float,
    n_samples: int,
) -> np.ndarray:
    """Bring a (possibly shorter) clean recording onto the artifact's grid.

    Resamples ``clean`` (``n_channels, n``) from ``src_sfreq`` to ``dst_sfreq``
    and tiles/trims it to exactly ``n_samples`` columns. Used for
    ``clean_source='niazy_pretrigger'``: the real pre-trigger segment (~28 s) is
    repeated to span the full artifact recording. Tiling introduces a few seams
    and a periodicity equal to the source length — acceptable for the
    single-recording proof-fit; a longer external corpus removes it (run_3 §3).
    """
    clean = np.asarray(clean, dtype=np.float64)
    if abs(src_sfreq - dst_sfreq) > 1e-6:
        g = gcd(int(round(dst_sfreq)), int(round(src_sfreq)))
        up, down = int(round(dst_sfreq)) // g, int(round(src_sfreq)) // g
        clean = resample_poly(clean, up, down, axis=-1)
    n = clean.shape[-1]
    if n == 0:
        raise ValueError("clean recording is empty")
    reps = int(np.ceil(n_samples / n))
    tiled = np.tile(clean, (1, reps))[:, :n_samples]
    return tiled.astype(np.float32)


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


def inject_real_ieds(
    clean: np.ndarray,
    clean_sfreq: float,
    ch_names: list[str],
    ied_pool: list[dict],
    pool_sfreq: float,
    *,
    rate_hz: float,
    amplitude_scale: float = 1.0,
    taper_s: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Inject **real** annotated IEDs (run_3 §6.6 / run_6 ground truth).

    Each pool entry is a real multi-channel spike-wave complex cut around a ``!``
    onset of the VEPISET IED dataset, **in its original volts**, with the
    originating 10-20 electrode names and the marker offset. Channels are mapped
    to the target montage **by name** (exact placement — both use the 10-20
    system), so the real morphology *and* the real cross-channel topography are
    preserved.

    This function performs **pure injection — no detection.** The IED is known
    ground truth (an expert marked it), so nothing about it is re-estimated:

    * **Amplitude is not rescaled.** VEPISET and the in-scanner recording are both
      scalp EEG in volts with a comparable background (~16 vs ~17 µV mean |.|), so
      a real IED already sits at a realistic level. Any renormalisation would
      falsify the event. ``amplitude_scale`` stays at 1.0 except for deliberate
      ablations.
    * **Background is removed, the spike is not.** Only the source window's
      baseline (offset + drift, fitted on the spike-free window edges) is
      subtracted and the edges are raised-cosine tapered over ``taper_s``, so the
      transient is added without importing VEPISET's own background or a step
      discontinuity.
    * **The label is the injected event.** Every mapped channel is labelled at the
      marker time. There is deliberately no visibility or sharpness gate: an IED
      momentarily masked by BCG is still an IED, and gating it out would teach the
      model that masked spikes may be deleted.

    Returns the spiked signal and the ``(channel, marker_sample)`` centers.
    """
    if not ied_pool:
        raise ValueError("inject_real_ieds requires a non-empty ied_pool")
    rng = np.random.default_rng(seed)
    out = clean.astype(np.float32).copy()
    n_channels, n_samples = out.shape
    name_to_idx = {_LEGACY_NAME_ALIAS.get(n, n): i for i, n in enumerate(ch_names)}
    name_to_idx.update({n: i for i, n in enumerate(ch_names)})  # also accept raw names
    duration_s = n_samples / float(clean_sfreq)
    n_spikes = int(rng.poisson(rate_hz * duration_s))
    centers: list[tuple[int, int]] = []

    for _ in range(n_spikes):
        ied = ied_pool[int(rng.integers(len(ied_pool)))]
        wf = np.asarray(ied["waveforms"], dtype=np.float64)  # (n_named, T) in volts
        names = list(ied["names"])
        t_pool = wf.shape[-1]
        marker_pool = int(ied.get("marker", t_pool // 2))    # '!' onset (defaults to centre)
        if abs(pool_sfreq - clean_sfreq) > 1e-6:
            g = gcd(int(round(clean_sfreq)), int(round(pool_sfreq)))
            wf = resample_poly(wf, int(round(clean_sfreq)) // g, int(round(pool_sfreq)) // g, axis=-1)
        t_len = wf.shape[-1]
        if t_len >= n_samples or t_len < 4:
            continue
        marker_r = min(t_len - 1, max(0, int(round(marker_pool * t_len / t_pool))))
        start = int(rng.integers(0, n_samples - t_len))
        edge = int(np.clip(round(taper_s * clean_sfreq), 1, t_len // 4))
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, edge + 1) / (edge + 1)))
        edge_idx = np.concatenate([np.arange(edge), np.arange(t_len - edge, t_len)])
        grid = np.arange(t_len, dtype=np.float64)
        for r, nm in enumerate(names):
            idx = name_to_idx.get(_LEGACY_NAME_ALIAS.get(nm, nm), name_to_idx.get(nm))
            if idx is None:
                continue
            wave = wf[r] * float(amplitude_scale)
            # Baseline (offset + drift) from the spike-free window edges, then taper
            # the edges — isolates the transient without touching its amplitude.
            slope, intercept = np.polyfit(edge_idx, wave[edge_idx], 1)
            wave = wave - (intercept + slope * grid)
            wave[:edge] *= ramp
            wave[t_len - edge:] *= ramp[::-1]
            out[idx, start:start + t_len] += wave.astype(np.float32)
            centers.append((idx, start + marker_r))
    return out.astype(np.float32), centers


# ---------------------------------------------------------------------------
# AAS failure modes (run_3 §2 — under-subtraction headroom)
# ---------------------------------------------------------------------------


def apply_artifact_failure_modes(
    artifact: np.ndarray,
    sfreq: float,
    starts: np.ndarray,
    stops: np.ndarray,
    *,
    epoch_amplitude_jitter: float = 0.03,
    epoch_timing_jitter_samples: float = 0.05,
    motion_drift_depth: float = 0.05,
    motion_drift_hz: float = 0.05,
    helium_pump_uv: float = 2.0,
    helium_pump_hz: float = 46.0,
    seed: int = 0,
) -> np.ndarray:
    """Add the artifact variability that AAS *cannot* average away.

    Without this, the target artifact is the AAS+OBS **template** — a stereotyped,
    epoch-invariant estimate. A model trained on it can at best reproduce AAS, so
    it inherits exactly AAS' in-band residual and has no route to beating it
    (run_3 §2). Superimposing the known AAS failure modes makes the synthetic
    artifact *richer than the template*, and because the enriched artifact is also
    the training target, the model is required to remove precisely what averaging
    leaves behind:

    * ``epoch_amplitude_jitter`` — per-epoch gain variation (sd, relative).
    * ``epoch_timing_jitter_samples`` — per-epoch sub-sample timing jitter (sd, in
      samples): the trigger/ADC clock mismatch behind AAS' sub-sample residual.
    * ``motion_drift_depth`` / ``motion_drift_hz`` — slow, per-channel amplitude
      modulation standing in for head motion inside the gradient field.
    * ``helium_pump_uv`` / ``helium_pump_hz`` — narrowband helium-pump line, an
      in-band contaminant that slice-averaging does not touch.

    Gains and shifts are interpolated *between epoch centres*, so the result is a
    smooth warp/modulation with no artificial discontinuity at epoch boundaries.
    Set any parameter to 0 to disable that mode.

    Returns a new artifact array; the input is not modified.
    """
    art = np.asarray(artifact, dtype=np.float64)
    n_channels, n_samples = art.shape
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64)
    centres = ((np.asarray(starts, dtype=np.float64) + np.asarray(stops, dtype=np.float64)) / 2.0)
    n_epochs = centres.size
    if n_epochs < 2:
        return art.astype(np.float32)

    gain_t = np.ones(n_samples, dtype=np.float64)
    if epoch_amplitude_jitter > 0:
        gain_t *= np.interp(t, centres, 1.0 + rng.normal(0.0, epoch_amplitude_jitter, n_epochs))
    shift_t = (
        np.interp(t, centres, rng.normal(0.0, epoch_timing_jitter_samples, n_epochs))
        if epoch_timing_jitter_samples > 0
        else None
    )

    out = np.empty_like(art)
    for c in range(n_channels):
        chan = art[c]
        # Sub-sample timing jitter as a smooth time warp (no epoch-edge steps).
        warped = np.interp(t + shift_t, t, chan) if shift_t is not None else chan
        g = gain_t
        if motion_drift_depth > 0:
            # Per-channel phase: motion modulates electrodes differently.
            g = g * (1.0 + motion_drift_depth * np.sin(2.0 * np.pi * motion_drift_hz * t / sfreq + rng.uniform(0, 2 * np.pi)))
        out[c] = warped * g
        if helium_pump_uv > 0:
            out[c] += (helium_pump_uv * 1e-6) * rng.uniform(0.5, 1.5) * np.sin(
                2.0 * np.pi * helium_pump_hz * t / sfreq + rng.uniform(0, 2 * np.pi)
            )
    return out.astype(np.float32)


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
    pretrigger_clean: np.ndarray | None = None,
    pretrigger_sfreq: float | None = None,
    inject_spikes_mode: bool = False,
    spike_source: str = "synthetic",
    real_ied_pool: list[dict] | None = None,
    real_ied_sfreq: float | None = None,
    spike_rate_hz: float = 0.7,
    spike_amplitude_uv: float = 40.0,
    spike_amplitude_scale: float = 1.0,
    spike_taper_s: float = 0.05,
    spike_width_ms: float = 20.0,
    spike_label_halfwidth: int = 3,
    val_fraction: float = 0.2,
    failure_modes: bool = True,
    epoch_amplitude_jitter: float = 0.03,
    epoch_timing_jitter_samples: float = 0.05,
    motion_drift_depth: float = 0.05,
    motion_drift_hz: float = 0.05,
    helium_pump_uv: float = 2.0,
    helium_pump_hz: float = 46.0,
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

    if spike_source not in ("synthetic", "real_ied"):
        raise ValueError(f"spike_source must be 'synthetic' or 'real_ied', got {spike_source!r}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    # --- epoch grid (needed before the clean, so the clean can be split too) ---
    starts, stops = epoch_boundaries(bundle["triggers"], sfreq, offset_seconds, n_samples)
    n_epochs = starts.size
    radius = context_epochs // 2
    if n_epochs <= 2 * radius:
        raise ValueError(f"Not enough epochs ({n_epochs}) for context width {context_epochs}")
    # Contiguous train/val boundary in epoch and sample space. Everything after it
    # is validation; a guard of `radius` epochs is dropped at the seam because the
    # 7-epoch contexts overlap (run_3 §7).
    split_epoch = max(radius + 1, int(round(n_epochs * (1.0 - val_fraction))))
    split_epoch = min(split_epoch, n_epochs - radius - 1)
    split_sample = int(starts[split_epoch])

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
    elif clean_source == "niazy_pretrigger":
        # Real in-scanner pre-trigger EEG: brain + BCG, GA-free (run_3 §3). Same
        # Niazy montage, so channels align 1:1; resample to the bundle rate and
        # tile to span the artifact recording. BCG rides along in situ.
        if pretrigger_clean is None or pretrigger_sfreq is None:
            raise ValueError("clean_source='niazy_pretrigger' requires pretrigger_clean and pretrigger_sfreq")
        pre = np.asarray(pretrigger_clean, dtype=np.float32)
        if pre.shape[0] != n_channels:
            raise ValueError(
                f"pretrigger_clean has {pre.shape[0]} channels but the artifact bundle has {n_channels}; "
                "both must use the same Niazy montage in the same order"
            )
        # The pre-trigger segment is far shorter than the recording, so it has to be
        # tiled — which would make the SAME clean appear on both sides of the
        # train/val boundary. Split the source segment disjointly first and tile each
        # part only within its own epoch range, so no validation background was ever
        # seen during training (run_3 §7).
        pre_cut = int(round(pre.shape[1] * (1.0 - val_fraction)))
        pre_cut = int(np.clip(pre_cut, 1, pre.shape[1] - 1))
        clean_true = np.empty((n_channels, n_samples), dtype=np.float32)
        clean_true[:, :split_sample] = resample_and_tile(
            pre[:, :pre_cut], float(pretrigger_sfreq), sfreq, split_sample
        )
        clean_true[:, split_sample:] = resample_and_tile(
            pre[:, pre_cut:], float(pretrigger_sfreq), sfreq, n_samples - split_sample
        )
    else:  # synthetic
        target_rms = float(np.sqrt(np.mean(corrected.astype(np.float64) ** 2))) or 1.0
        clean_true = synthetic_clean(n_channels, n_samples, sfreq, target_rms=target_rms, seed=seed)

    # --- spike injection, decoupled from the clean source (run_3 §6.6) ---
    # Real IEDs (or synthetic) can be injected onto ANY clean — in particular the
    # real niazy_pretrigger clean — so the spike-preservation ground truth (run_6)
    # rides on a realistic background.
    spikes_enabled = bool(inject_spikes_mode)
    if spikes_enabled:
        if spike_source == "real_ied":
            if not real_ied_pool:
                raise ValueError("spike_source='real_ied' requires real_ied_pool")
            clean_true, spike_centers = inject_real_ieds(
                clean_true, sfreq, ch_names, real_ied_pool, float(real_ied_sfreq or sfreq),
                rate_hz=spike_rate_hz,
                amplitude_scale=spike_amplitude_scale,
                taper_s=spike_taper_s,
                seed=seed + 1,
            )
        else:  # synthetic
            clean_true, spike_centers = inject_spikes(
                clean_true, sfreq, rate_hz=spike_rate_hz,
                amplitude=spike_amplitude_uv * 1e-6, width_ms=spike_width_ms, seed=seed + 1,
            )

    spikes_by_channel: dict[int, np.ndarray] = {}
    if spike_centers:
        for ch, sample in spike_centers:
            spikes_by_channel.setdefault(ch, []).append(sample)
        spikes_by_channel = {ch: np.asarray(v, dtype=np.int64) for ch, v in spikes_by_channel.items()}

    # --- guard geometry ---
    out_len = core_samples + 2 * guard_samples
    neighbors = select_neighbors(ch_names, k_neighbors=k_neighbors)
    eeg_channels = list(range(n_channels))

    # --- AAS failure modes (run_3 §2): enrich the artifact beyond the template so
    # the model has under-subtraction headroom. Applied to `artifact`, which is
    # both the summand of `noisy` and the training target, so what averaging
    # cannot remove becomes something the model is required to remove.
    # Keep the pre-enrichment template: it is what an *ideal* AAS would remove
    # (the epoch-repeatable component), so run_6 can compare the model against
    # AAS on identical data without re-running the pipeline.
    artifact_template = artifact
    if failure_modes:
        artifact = apply_artifact_failure_modes(
            artifact, sfreq, starts, stops,
            epoch_amplitude_jitter=epoch_amplitude_jitter,
            epoch_timing_jitter_samples=epoch_timing_jitter_samples,
            motion_drift_depth=motion_drift_depth,
            motion_drift_hz=motion_drift_hz,
            helium_pump_uv=helium_pump_uv,
            helium_pump_hz=helium_pump_hz,
            seed=seed + 2,
        )

    clean_ex: list[np.ndarray] = []
    artifact_ex: list[np.ndarray] = []
    target_ex: list[np.ndarray] = []
    template_ex: list[np.ndarray] = []
    template_ctx_ex: list[np.ndarray] = []
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
    split_ex: list[int] = []
    n_guard_dropped = 0
    for center_idx in range(radius, n_epochs - radius):
        # Leakage-free assignment: an example belongs to train only if its WHOLE
        # 7-epoch context lies before the boundary, to val only if it lies wholly
        # after. Contexts straddling the seam are dropped (run_3 §7).
        if center_idx + radius < split_epoch:
            example_split = 0
        elif center_idx - radius >= split_epoch:
            example_split = 1
        else:
            n_guard_dropped += len(eeg_channels)
            continue
        for c in eeg_channels:
            chans = neighbors[c]  # [c, n1, ..., nk]
            clean_block = _ctx_block(clean_true, chans, center_idx)
            artifact_block = _ctx_block(artifact, chans, center_idx)
            # Template over the WHOLE context, not just the centre epoch: the
            # residual formulation feeds the model `noisy - template`, so it needs
            # the AAS/FARM-removable part of every input epoch and channel.
            template_block = _ctx_block(artifact_template, chans, center_idx)

            clean_ex.append(clean_block)
            artifact_ex.append(artifact_block)
            target_ex.append(artifact_block[radius, 0:1].copy())  # artifact of (c, center epoch)
            # AAS-removable part of the same epoch (run_6 Phase E reference)
            template_ex.append(
                _guard_window(
                    artifact_template[int(c)],
                    int(starts[center_idx]),
                    int(stops[center_idx]),
                    int(round(guard_samples * (int(stops[center_idx]) - int(starts[center_idx])) / core_samples)),
                    out_len,
                )[None, :].copy()
            )
            spike_ex.append(_spike_mask(c, center_idx) if spikes_enabled else np.zeros((1, out_len), np.float32))
            template_ctx_ex.append(template_block)
            neigh_ex.append(chans.copy())
            tgt_ch_ex.append(int(c))
            center_ep_ex.append(int(center_idx))
            split_ex.append(example_split)

            if max_examples is not None and len(clean_ex) >= max_examples:
                stop = True
                break
        if stop:
            break
    if n_guard_dropped:
        logger.info(
            "Dropped %d examples in the train/val guard band (%d epochs at the seam)",
            n_guard_dropped, 2 * radius,
        )

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
        # What an ideal AAS removes: the epoch-repeatable template, before the
        # failure modes were superimposed. run_6 compares against this.
        "artifact_center_template": np.stack(template_ex, axis=0).astype(np.float32),
        "artifact_context_template": np.stack(template_ctx_ex, axis=0).astype(np.float32),
        "spike_labels": spike_labels.astype(np.float32),
        "neighbor_channel_indices": np.stack(neigh_ex, axis=0).astype(np.int64),
        "target_channel_index": np.asarray(tgt_ch_ex, dtype=np.int64),
        "center_epoch_index": np.asarray(center_ep_ex, dtype=np.int64),
        # 0 = train, 1 = validation. Precomputed here because only the builder knows
        # the clean partition and the context overlap; the dataset must honour it
        # rather than re-splitting at random (run_3 §7).
        "example_split": np.asarray(split_ex, dtype=np.int64),
        "split_epoch": np.asarray([split_epoch], dtype=np.int64),
        "val_fraction": np.asarray([val_fraction], dtype=np.float64),
        "failure_modes": np.asarray([bool(failure_modes)], dtype=bool),
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
