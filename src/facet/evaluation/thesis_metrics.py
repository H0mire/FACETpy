"""Reference-based metrics used by the thesis unified-holdout protocol."""

from __future__ import annotations

import numpy as np

EPS = 1e-20


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a - b)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _snr_db(reference: np.ndarray, error: np.ndarray) -> float:
    return float(10.0 * np.log10((np.mean(np.square(reference)) + EPS) / (np.mean(np.square(error)) + EPS)))


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def compute_metrics(
    noisy: np.ndarray,
    clean: np.ndarray,
    artifact: np.ndarray,
    pred_artifact: np.ndarray,
    *,
    sfreq_hz: float,
) -> dict[str, float | int | bool]:
    """Compute the canonical metric set on (N, 30, 512) arrays.

    `corrected = noisy - pred_artifact` (the artifact-subtraction contract).
    Result fields match the existing per-model evaluate.py outputs so the
    flat_metrics dict is directly comparable to Run 1 artefacts.
    """
    if not (noisy.shape == clean.shape == artifact.shape == pred_artifact.shape):
        raise ValueError(
            f"shape mismatch: noisy={noisy.shape} clean={clean.shape} "
            f"artifact={artifact.shape} pred={pred_artifact.shape}"
        )

    corrected = noisy - pred_artifact
    before_error = noisy - clean
    after_error = corrected - clean

    # FACETpy-style RMS recovery (target = 1.0): does the corrected signal
    # preserve the same amplitude statistics as the ground-truth clean?
    # Per-(window, channel) std along the time axis, then average over
    # window×channel pairs — mirrors RMSResidualCalculator at
    # src/facet/evaluation/metrics.py:1071-1153 (which computes std-per-channel,
    # then mean across channels on whole-recording data).
    std_corrected = np.std(corrected, axis=-1)  # (N, C)
    std_clean = np.maximum(np.std(clean, axis=-1), EPS)
    std_noisy = np.maximum(np.std(noisy, axis=-1), EPS)
    rms_recovery_ratio = float(np.mean(std_corrected / std_clean))
    rms_recovery_distance = float(np.mean(np.abs(std_corrected / std_clean - 1.0)))
    rms_baseline_noisy_ratio = float(np.mean(std_noisy / std_clean))

    metrics: dict[str, float | int | bool] = {
        "n_examples": int(noisy.shape[0]),
        "n_channels": int(noisy.shape[1]),
        "samples_per_epoch": int(noisy.shape[-1]),
        "sfreq_hz": float(sfreq_hz),
        "clean_mse_before": _mse(noisy, clean),
        "clean_mse_after": _mse(corrected, clean),
        "clean_mae_before": _mae(noisy, clean),
        "clean_mae_after": _mae(corrected, clean),
        "clean_snr_db_before": _snr_db(clean, before_error),
        "clean_snr_db_after": _snr_db(clean, after_error),
        "artifact_mse": _mse(pred_artifact, artifact),
        "artifact_mae": _mae(pred_artifact, artifact),
        "artifact_corr": _corrcoef(pred_artifact, artifact),
        "artifact_snr_db": _snr_db(artifact, pred_artifact - artifact),
        "residual_error_rms_ratio": _rms(after_error) / (_rms(before_error) + EPS),
        # FACETpy-style amplitude metrics (target = 1.0 for perfect preservation)
        "rms_recovery_ratio": rms_recovery_ratio,
        "rms_recovery_distance": rms_recovery_distance,
        "rms_baseline_noisy_ratio": rms_baseline_noisy_ratio,
    }
    metrics["clean_mse_reduction_pct"] = 100.0 * (
        1.0 - metrics["clean_mse_after"] / (metrics["clean_mse_before"] + EPS)
    )
    metrics["clean_snr_improvement_db"] = metrics["clean_snr_db_after"] - metrics["clean_snr_db_before"]
    return metrics
