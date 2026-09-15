"""Spike-preservation metrics for Run 6 (``docs/research/run_6_...``, Phase D).

The clinically interesting question is *not* whether a correction leaves the
spike amplitude intact. AAS subtracts only the epoch-repeatable component, and an
interictal discharge is not phase-locked to the MR trigger, so it barely enters
the template at all — both AAS and a DL model return a spike close to its
original height. What separates them is what surrounds the spike: AAS leaves the
non-repeatable residual (jitter, motion modulation, helium pump) standing, and
those residuals are transient, broadband and sharp-edged — i.e. they look like
spikes. That is what produces false positives in clinical review.

The headline metrics are therefore ``spike_neighborhood_snr_db`` and
``spike_contrast_db``; amplitude ratio and morphology correlation are guard-rails
that catch the DL-specific failure of *learning the spike away*.

All functions take ``(N, T)`` arrays of one channel each and a boolean spike
mask, and work in volts.
"""

from __future__ import annotations

import numpy as np


def expand_mask(mask: np.ndarray, margin_samples: int) -> np.ndarray:
    """Widen a boolean mask by ``margin_samples`` in both directions."""
    if margin_samples <= 0:
        return mask.astype(bool)
    out = mask.astype(bool).copy()
    for shift in range(1, int(margin_samples) + 1):
        out[:, shift:] |= mask[:, :-shift].astype(bool)
        out[:, :-shift] |= mask[:, shift:].astype(bool)
    return out


def _snr_db(signal: np.ndarray, error: np.ndarray) -> float:
    """10·log10(power(signal) / power(error)); NaN when either side is empty."""
    if signal.size == 0 or error.size == 0:
        return float("nan")
    sig_p = float(np.mean(np.asarray(signal, dtype=np.float64) ** 2))
    err_p = float(np.mean(np.asarray(error, dtype=np.float64) ** 2))
    if err_p <= 0:
        return float("inf")
    if sig_p <= 0:
        return float("nan")
    return float(10.0 * np.log10(sig_p / err_p))


def compute_spike_metrics_per_example(
    pred_clean: np.ndarray,
    target_clean: np.ndarray,
    spike_labels: np.ndarray,
    *,
    neighborhood_samples: int = 200,
) -> dict[str, np.ndarray]:
    """Per-spike-example values behind :func:`compute_spike_metrics`' averages.

    Averages cannot support a paired comparison. Two corrections evaluated on the
    same examples differ *per example*, and only those differences give a paired
    test, an effect size and an interval — which is what the results protocol
    requires for any "A beats B" statement. Returned arrays are aligned:
    ``example_index[k]`` identifies the row every other array's ``k``-th entry
    came from, so two methods' outputs can be matched without re-deriving the
    selection.

    Entries are NaN where a metric is undefined for that example (flat target
    window, residual-free ring); callers drop them pairwise and report the drop.
    """
    pred = np.asarray(pred_clean, dtype=np.float64)
    target = np.asarray(target_clean, dtype=np.float64)
    spike = np.asarray(spike_labels) > 0
    if not (pred.shape == target.shape == spike.shape):
        raise ValueError(f"shapes must match, got {pred.shape}, {target.shape}, {spike.shape}")

    neighborhood = expand_mask(spike, neighborhood_samples)
    ring = neighborhood & ~spike
    err = pred - target
    rows = np.flatnonzero(spike.any(axis=1))

    keys = (
        "amplitude_ratio",
        "morphology_corr",
        "contrast_db",
        "latency_drift_samples",
        "neighborhood_snr_db",
        "peak_over_residual",
        "rmse_uv",
    )
    out: dict[str, np.ndarray] = {k: np.full(rows.size, np.nan) for k in keys}
    out["example_index"] = rows.astype(np.int64)

    for k, i in enumerate(rows):
        idx = np.flatnonzero(spike[i])
        lo, hi = int(idx.min()), int(idx.max()) + 1
        t_win, p_win = target[i, lo:hi], pred[i, lo:hi]
        t_peak, p_peak = float(np.max(np.abs(t_win))), float(np.max(np.abs(p_win)))
        ring_i = ring[i]
        resid = float(np.sqrt(np.mean(err[i, ring_i] ** 2))) if ring_i.any() else 0.0

        if t_peak > 0:
            out["amplitude_ratio"][k] = p_peak / t_peak
            if resid > 0:
                # A completely deleted spike has p_peak == 0, whose contrast is
                # minus infinity. Computing it as log10(0) is arithmetically the
                # same answer but raises a divide-by-zero warning, which under a
                # warnings-as-errors test run is a crash rather than a result.
                out["contrast_db"][k] = (20.0 * np.log10(p_peak / resid)
                                         if p_peak > 0 else float("-inf"))
                out["peak_over_residual"][k] = t_peak / resid
        out["latency_drift_samples"][k] = float(np.argmax(np.abs(p_win)) - np.argmax(np.abs(t_win)))
        if t_win.size > 1 and np.std(t_win) > 0 and np.std(p_win) > 0:
            out["morphology_corr"][k] = float(np.corrcoef(t_win, p_win)[0, 1])
        if ring_i.any():
            out["neighborhood_snr_db"][k] = _snr_db(target[i, ring_i], err[i, ring_i])
        out["rmse_uv"][k] = float(np.sqrt(np.mean(err[i] ** 2))) * 1e6
    return out


def compute_spike_metrics(
    pred_clean: np.ndarray,
    target_clean: np.ndarray,
    spike_labels: np.ndarray,
    *,
    neighborhood_samples: int = 200,
) -> dict[str, float]:
    """Compare a corrected signal against the true clean around known spikes.

    Parameters
    ----------
    pred_clean, target_clean : np.ndarray, shape ``(N, T)``
        Corrected estimate and ground-truth clean, in volts.
    spike_labels : np.ndarray, shape ``(N, T)``
        Non-zero where a ground-truth IED was injected.
    neighborhood_samples : int
        Half-width of the ring analysed around each spike.

    Returns
    -------
    dict
        See module docstring; ``spike_neighborhood_snr_db`` and
        ``spike_contrast_db`` are the headline numbers.
    """
    pred = np.asarray(pred_clean, dtype=np.float64)
    target = np.asarray(target_clean, dtype=np.float64)
    spike = np.asarray(spike_labels) > 0
    if not (pred.shape == target.shape == spike.shape):
        raise ValueError(f"shapes must match, got {pred.shape}, {target.shape}, {spike.shape}")

    has_spike = spike.any(axis=1)
    neighborhood = expand_mask(spike, neighborhood_samples)
    ring = neighborhood & ~spike            # around the spike, excluding it
    non_spike = ~spike

    err = pred - target
    out: dict[str, float] = {
        "spike_neighborhood_snr_db": _snr_db(target[ring], err[ring]),
        "non_spike_snr_db": _snr_db(target[non_spike], err[non_spike]),
    }

    amp_ratios: list[float] = []
    latency_ms: list[float] = []
    morph_corr: list[float] = []
    contrast_db: list[float] = []
    for i in np.flatnonzero(has_spike):
        idx = np.flatnonzero(spike[i])
        lo, hi = int(idx.min()), int(idx.max()) + 1
        t_win, p_win = target[i, lo:hi], pred[i, lo:hi]

        t_peak = float(np.max(np.abs(t_win)))
        p_peak = float(np.max(np.abs(p_win)))
        if t_peak > 0:
            amp_ratios.append(p_peak / t_peak)
            # Contrast: how far the spike stands out of the residual left behind
            # in its own neighbourhood — the actual detectability measure.
            ring_i = ring[i]
            if ring_i.any():
                resid = float(np.sqrt(np.mean(err[i, ring_i] ** 2)))
                # A residual-free neighbourhood means unbounded contrast; record it
                # as such rather than dropping the spike from the average, which
                # would silently make a perfect reconstruction look unmeasurable.
                if resid <= 0:
                    contrast_db.append(float("inf"))
                elif p_peak <= 0:
                    contrast_db.append(float("-inf"))    # spike deleted; see above
                else:
                    contrast_db.append(20.0 * np.log10(p_peak / resid))
        latency_ms.append(float(np.argmax(np.abs(p_win)) - np.argmax(np.abs(t_win))))
        if t_win.size > 1 and np.std(t_win) > 0 and np.std(p_win) > 0:
            morph_corr.append(float(np.corrcoef(t_win, p_win)[0, 1]))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    # Regime check. Every spike-level metric silently degenerates once the
    # leftover residual is bigger than the spike itself: max|pred| in the spike
    # window then measures residual, not spike, and the amplitude ratio blows up
    # past 1 instead of falling below it. Report the ratio of spike peak to local
    # residual so the reader can see whether the spike metrics mean anything.
    peak_over_resid: list[float] = []
    for i in np.flatnonzero(has_spike):
        ring_i = ring[i]
        if not ring_i.any():
            continue
        resid = float(np.sqrt(np.mean(err[i, ring_i] ** 2)))
        t_peak = float(np.max(np.abs(target[i, spike[i]])))
        if resid > 0 and t_peak > 0:
            peak_over_resid.append(t_peak / resid)
    out["spike_peak_over_residual"] = _mean(peak_over_resid)
    out["spike_metrics_reliable"] = float(out["spike_peak_over_residual"] >= 1.0)

    out["spike_amplitude_ratio"] = _mean(amp_ratios)
    out["spike_contrast_db"] = _mean(contrast_db)
    out["spike_morphology_corr"] = _mean(morph_corr)
    out["spike_peak_latency_drift_samples"] = _mean(latency_ms)
    out["n_spike_examples"] = float(int(has_spike.sum()))
    return out
