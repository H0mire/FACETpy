"""Honesty metrics: did the model compute the *real* artifact?

The core failure mode of an artifact-removal model is **plausible fabrication**:
it subtracts just enough so the output *looks like* EEG, without recovering the
*specific* true signal. RMSE/SNR can be gamed by such a model (a smoothed or
shrunk output scores well); correlation against the ground truth cannot — two
independent EEG-like signals correlate near zero.

These metrics require ground truth, which the semi-synthetic Weg-A dataset has
(``noisy = clean + artifact`` with both known). They are computed on plain numpy
arrays so they can be called from any training/eval loop.

Key quantities (``removed = noisy - output``, ``true_artifact = noisy - clean``,
``residual = output - clean``):

* ``artifact_rms_reduction`` — ``1 - rms(residual)/rms(true_artifact)``. **The
  decisive "did it compute the artifact" number** (~1 = removed the artifact's
  energy accurately). Robust even where the artifact dominates the signal.
* ``clean_corr`` — corr(output, clean). Does the output match the *specific* true
  clean? A generic EEG-like fabrication scores ~0 — this rules out "looks like
  EEG but isn't the true one".
* ``residual_vs_clean_corr`` — corr(residual, clean). Should be ~0. If high, real
  brain signal leaked into the error (distortion/fabrication).
* ``residual_vs_artifact_corr`` — corr(residual, true_artifact). Should be ~0. If
  high, the model systematically over-/under-subtracted the artifact.

``artifact_corr`` = corr(removed, true_artifact) is also reported, but it is a
**weak** discriminator when the artifact is much larger than the signal: then
``removed ≈ noisy ≈ artifact`` regardless of the model, so it stays high even for
a bad one. Gate on ``artifact_rms_reduction`` + ``clean_corr`` instead (see
``honesty_verdict``).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _flat(a), _flat(b)
    if a.size != b.size or a.size < 2:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < EPS or sb < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(_flat(x)))))


def honesty_report(noisy: np.ndarray, output: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    """Compute the honesty metrics for a denoised ``output`` against ground truth.

    Parameters
    ----------
    noisy, output, clean : np.ndarray
        The corrupted input, the model's denoised output, and the true clean
        signal. Same shape; broadcast/flattened internally. Pass ``true_artifact``
        models' reconstructed clean here (``output``), not the artifact estimate.

    Returns
    -------
    dict
        ``artifact_corr``, ``clean_corr``, ``residual_rms_ratio``,
        ``residual_vs_clean_corr``, ``residual_vs_artifact_corr``,
        ``artifact_rms_reduction``.
    """
    noisy, output, clean = _flat(noisy), _flat(output), _flat(clean)
    removed = noisy - output
    true_artifact = noisy - clean
    residual = output - clean
    return {
        "artifact_corr": _corr(removed, true_artifact),
        "clean_corr": _corr(output, clean),
        "residual_rms_ratio": _rms(residual) / (_rms(clean) + EPS),
        "residual_vs_clean_corr": _corr(residual, clean),
        "residual_vs_artifact_corr": _corr(residual, true_artifact),
        "artifact_rms_reduction": 1.0 - _rms(residual) / (_rms(true_artifact) + EPS),
    }


def honesty_verdict(
    report: dict[str, float],
    *,
    min_artifact_rms_reduction: float = 0.9,
    min_clean_corr: float = 0.9,
    max_residual_leak_corr: float = 0.2,
) -> dict[str, object]:
    """Heuristic pass/fail on top of :func:`honesty_report`.

    Gates on the *robust* discriminators (not ``artifact_corr``): the model must
    remove most of the true artifact's energy (``artifact_rms_reduction``),
    recover the *specific* clean (``clean_corr``), and leave little real signal in
    the residual. Defaults are deliberately strict — guidance, not a universal
    law; the right thresholds depend on the data and the reference's reliability.
    """
    leak = max(abs(report.get("residual_vs_clean_corr", 0.0)), abs(report.get("residual_vs_artifact_corr", 0.0)))
    recovers_artifact = report.get("artifact_rms_reduction", 0.0) >= min_artifact_rms_reduction
    matches_clean = report.get("clean_corr", 0.0) >= min_clean_corr
    structureless_residual = leak <= max_residual_leak_corr
    return {
        "recovers_real_artifact": bool(recovers_artifact),
        "matches_specific_clean": bool(matches_clean),
        "residual_structureless": bool(structureless_residual),
        "honest": bool(recovers_artifact and matches_clean and structureless_residual),
        "max_residual_leak_corr": float(leak),
    }
