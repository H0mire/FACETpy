"""Numeric validation of the spike-preservation metrics against closed forms.

The existing suite in ``test_spike_metrics.py`` covers shapes, edge cases and
degenerate inputs. That leaves the question a results chapter actually depends
on unanswered: **does each metric return the number it claims to return?**

Every test here builds an input whose expected value can be written down by
hand — a spike scaled by a known factor, shifted by a known number of samples,
or an error of a known constant amplitude — and asserts the metric reproduces
it to floating-point tolerance. A metric that drifts from its definition then
fails here rather than silently changing a thesis table.

Layout of the synthetic signals: one row, ``T`` samples, a spike label spanning
a known index range, and a neighbourhood ring whose width follows from
``neighborhood_samples``.
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.training.spike_metrics import (
    compute_spike_metrics,
    compute_spike_metrics_per_example,
    expand_mask,
)

T = 400
SPIKE_LO, SPIKE_HI = 200, 207  # label spans [200, 207)
MARGIN = 20  # ring half-width in samples


def _labels(lo: int = SPIKE_LO, hi: int = SPIKE_HI, n_rows: int = 1) -> np.ndarray:
    lab = np.zeros((n_rows, T), dtype=np.float32)
    lab[:, lo:hi] = 1.0
    return lab


def _triangular_spike(peak_uv: float, lo: int = SPIKE_LO, hi: int = SPIKE_HI) -> np.ndarray:
    """A single asymmetric spike in volts, peaking at the centre of the label.

    Asymmetric on purpose: a symmetric shape would make the latency test pass
    for the wrong reason, because argmax of a symmetric window is ambiguous.
    The peak sits at the window centre so a small shift stays inside the label —
    see :func:`test_latency_drift_is_truncated_at_the_label_edge` for what
    happens when it does not.
    """
    sig = np.zeros((1, T), dtype=np.float64)
    n = hi - lo
    mid = n // 2
    shape = np.concatenate(
        [
            np.linspace(0.0, 0.9, mid, endpoint=False),  # slow rise
            [1.0],  # peak, at the centre
            np.linspace(0.45, 0.0, n - mid - 1),  # faster fall
        ]
    )
    sig[0, lo:hi] = shape * peak_uv * 1e-6
    return sig


# --------------------------------------------------------------------- mask


@pytest.mark.unit
def test_expand_mask_widens_by_exactly_the_margin():
    mask = np.zeros((1, 21), dtype=bool)
    mask[0, 10] = True
    out = expand_mask(mask, 3)
    assert out[0, 7:14].all()
    assert not out[0, :7].any()
    assert not out[0, 14:].any()
    assert out.sum() == 7  # 2*3 + 1


@pytest.mark.unit
def test_expand_mask_zero_margin_is_identity():
    mask = np.zeros((1, 10), dtype=bool)
    mask[0, 4] = True
    assert np.array_equal(expand_mask(mask, 0), mask)


# ---------------------------------------------------------- amplitude ratio


@pytest.mark.unit
@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 1.5, 3.0])
def test_amplitude_ratio_equals_the_applied_scale(scale):
    """pred = scale * target inside the spike window -> ratio is exactly scale."""
    target = _triangular_spike(50.0)
    pred = target * scale
    out = compute_spike_metrics_per_example(pred, target, _labels(), neighborhood_samples=MARGIN)
    assert out["amplitude_ratio"][0] == pytest.approx(scale, rel=1e-12)


@pytest.mark.unit
def test_amplitude_ratio_of_a_deleted_spike_is_zero():
    target = _triangular_spike(50.0)
    out = compute_spike_metrics_per_example(np.zeros_like(target), target, _labels(), neighborhood_samples=MARGIN)
    assert out["amplitude_ratio"][0] == pytest.approx(0.0, abs=1e-15)


# ------------------------------------------------------------- morphology


@pytest.mark.unit
def test_morphology_corr_is_one_for_a_positive_affine_prediction():
    """Correlation is scale- and offset-invariant, so a*x+b with a>0 gives r=1."""
    target = _triangular_spike(50.0)
    pred = target * 2.5 + 7e-6
    out = compute_spike_metrics_per_example(pred, target, _labels(), neighborhood_samples=MARGIN)
    assert out["morphology_corr"][0] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.unit
def test_morphology_corr_is_minus_one_for_an_inverted_prediction():
    target = _triangular_spike(50.0)
    out = compute_spike_metrics_per_example(-target, target, _labels(), neighborhood_samples=MARGIN)
    assert out["morphology_corr"][0] == pytest.approx(-1.0, rel=1e-12)


@pytest.mark.unit
def test_morphology_corr_matches_numpy_on_an_arbitrary_prediction():
    rng = np.random.default_rng(0)
    target = _triangular_spike(50.0)
    pred = target + rng.normal(0.0, 5e-6, target.shape)
    out = compute_spike_metrics_per_example(pred, target, _labels(), neighborhood_samples=MARGIN)
    expected = np.corrcoef(target[0, SPIKE_LO:SPIKE_HI], pred[0, SPIKE_LO:SPIKE_HI])[0, 1]
    assert out["morphology_corr"][0] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------- latency


@pytest.mark.unit
@pytest.mark.parametrize("shift", [-3, -2, -1, 0, 1, 2, 3])
def test_latency_drift_equals_the_applied_shift(shift):
    """Moving the predicted peak by k samples must report a drift of exactly k."""
    target = _triangular_spike(50.0)
    pred = np.roll(target, shift, axis=1)
    out = compute_spike_metrics_per_example(pred, target, _labels(), neighborhood_samples=MARGIN)
    assert out["latency_drift_samples"][0] == pytest.approx(float(shift))


@pytest.mark.unit
def test_latency_drift_is_truncated_at_the_label_edge():
    """A drift larger than the label half-width cannot be observed.

    ``argmax`` runs over the labelled span only, so once the predicted peak
    leaves that span the reported drift saturates at the edge. This is a real
    limit of the metric — with a 7-sample label the largest observable drift is
    3 samples in either direction — and it is asserted here so a results table
    is never read as if larger drifts had been measured and found small.
    """
    target = _triangular_spike(50.0)
    far = np.roll(target, 10, axis=1)
    out = compute_spike_metrics_per_example(far, target, _labels(), neighborhood_samples=MARGIN)
    half = (SPIKE_HI - SPIKE_LO) // 2
    assert abs(out["latency_drift_samples"][0]) <= half


# -------------------------------------------------------------------- SNR


@pytest.mark.unit
@pytest.mark.parametrize("ratio_db", [0.0, 6.0206, 20.0, -10.0])
def test_neighborhood_snr_matches_the_constructed_power_ratio(ratio_db):
    """Constant signal A and constant error E in the ring -> SNR = 20*log10(A/E)."""
    amp = 10e-6
    err_amp = amp / (10.0 ** (ratio_db / 20.0))
    labels = _labels()
    ring = expand_mask(labels > 0, MARGIN) & ~(labels > 0)

    target = np.zeros((1, T))
    target[ring] = amp  # constant magnitude in the ring
    target[0, SPIKE_LO:SPIKE_HI] = 50e-6  # spike itself, excluded from the ring
    pred = target.copy()
    pred[ring] = amp + err_amp  # constant error of known size

    out = compute_spike_metrics_per_example(pred, target, labels, neighborhood_samples=MARGIN)
    assert out["neighborhood_snr_db"][0] == pytest.approx(ratio_db, abs=1e-9)


@pytest.mark.unit
def test_perfect_reconstruction_gives_infinite_snr():
    target = _triangular_spike(50.0)
    target[0, :] += 3e-6
    out = compute_spike_metrics(target.copy(), target, _labels(), neighborhood_samples=MARGIN)
    assert np.isinf(out["spike_neighborhood_snr_db"])


# --------------------------------------------------------------- contrast


@pytest.mark.unit
def test_contrast_db_equals_peak_over_ring_rms_in_decibels():
    labels = _labels()
    ring = expand_mask(labels > 0, MARGIN) & ~(labels > 0)
    peak, err_amp = 50e-6, 5e-6

    target = _triangular_spike(peak * 1e6)
    pred = target.copy()
    pred[ring] += err_amp  # constant error -> ring RMS is err_amp

    out = compute_spike_metrics_per_example(pred, target, labels, neighborhood_samples=MARGIN)
    expected = 20.0 * np.log10(peak / err_amp)
    assert out["contrast_db"][0] == pytest.approx(expected, rel=1e-9)
    assert out["peak_over_residual"][0] == pytest.approx(peak / err_amp, rel=1e-9)


# ------------------------------------------------------------------- RMSE


@pytest.mark.unit
@pytest.mark.parametrize("err_uv", [0.5, 2.0, 17.25])
def test_rmse_of_a_constant_error_is_that_error_in_microvolts(err_uv):
    target = _triangular_spike(50.0)
    pred = target + err_uv * 1e-6
    out = compute_spike_metrics_per_example(pred, target, _labels(), neighborhood_samples=MARGIN)
    assert out["rmse_uv"][0] == pytest.approx(err_uv, rel=1e-12)


# ------------------------------------------------- aggregate vs per-example


@pytest.mark.unit
def test_aggregate_amplitude_ratio_is_the_mean_of_the_per_example_values():
    """The summary must be the mean of the per-example values it summarises."""
    rows = 4
    target = np.zeros((rows, T))
    pred = np.zeros((rows, T))
    scales = [0.5, 1.0, 1.5, 2.0]
    for i, s in enumerate(scales):
        spike = _triangular_spike(40.0 + 5 * i)[0]
        target[i] = spike
        pred[i] = spike * s
    labels = _labels(n_rows=rows)

    per = compute_spike_metrics_per_example(pred, target, labels, neighborhood_samples=MARGIN)
    agg = compute_spike_metrics(pred, target, labels, neighborhood_samples=MARGIN)
    assert np.allclose(np.sort(per["amplitude_ratio"]), np.sort(scales), rtol=1e-12)
    assert agg["spike_amplitude_ratio"] == pytest.approx(float(np.mean(scales)), rel=1e-12)


@pytest.mark.unit
def test_a_dilated_label_scores_a_wider_window():
    """Widening the label must change what the morphology metric looks at.

    This underpins the label-width sensitivity analysis: the metric scores the
    span between the first and last labelled sample, so dilating the label is a
    real change of the scored extent, not a no-op.
    """
    target = _triangular_spike(50.0)
    pred = target.copy()
    pred[0, SPIKE_HI : SPIKE_HI + 30] += 20e-6  # damage only outside the narrow label

    narrow = compute_spike_metrics_per_example(pred, target, _labels(), neighborhood_samples=MARGIN)
    wide_labels = expand_mask(_labels() > 0, 30).astype(np.float32)
    wide = compute_spike_metrics_per_example(pred, target, wide_labels, neighborhood_samples=MARGIN)

    assert narrow["morphology_corr"][0] == pytest.approx(1.0, rel=1e-12)
    assert wide["morphology_corr"][0] < 0.999


@pytest.mark.unit
def test_a_fully_deleted_spike_yields_minus_infinite_contrast_without_warning():
    """Contrast of a deleted spike is -inf, and must be produced arithmetically.

    Computing it as ``log10(0)`` gives the same value but raises a numpy
    divide-by-zero warning. Under a warnings-as-errors test run that turns a
    legitimate result into a crash, so the metric special-cases it.
    """
    labels = _labels()
    ring = expand_mask(labels > 0, MARGIN) & ~(labels > 0)
    target = _triangular_spike(50.0)
    pred = np.zeros_like(target)
    pred[ring] = 3e-6  # non-zero residual, zero spike

    with np.errstate(all="raise"):
        out = compute_spike_metrics_per_example(pred, target, labels, neighborhood_samples=MARGIN)
        agg = compute_spike_metrics(pred, target, labels, neighborhood_samples=MARGIN)
    assert out["contrast_db"][0] == float("-inf")
    assert np.isneginf(agg["spike_contrast_db"])
