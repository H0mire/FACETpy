"""Closed-form tests for the deployment metrics.

Each case is built so the expected value can be derived by hand, not read off a
previous run. A metric that is only ever compared against its own last output
cannot catch the day it starts measuring something else.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.evaluation import (
    EpochSeamStepCalculator,
    GradientArtifactResidualCalculator,
)
from facet.evaluation.deployment_metrics import _epoch_rate_hz

SFREQ = 1000.0
PERIOD = 100          # samples per epoch -> 10 Hz repetition rate
N_EPOCHS = 60
N_CHANNELS = 4


def _context(data: np.ndarray, triggers: np.ndarray) -> ProcessingContext:
    info = mne.create_info([f"EEG{i:03d}" for i in range(data.shape[0])], SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    meta = ProcessingMetadata(triggers=list(triggers.astype(int)))
    return ProcessingContext(raw=raw, metadata=meta)


def _triggers() -> np.ndarray:
    return np.arange(N_EPOCHS) * PERIOD


@pytest.mark.unit
def test_epoch_rate_from_median_spacing():
    """The rate comes from the median gap, so one bad trigger cannot move it."""
    trg = _triggers().astype(np.int64)
    assert _epoch_rate_hz(trg, SFREQ) == pytest.approx(10.0)

    broken = np.delete(trg, 10)                     # one missing trigger -> one 2x gap
    assert _epoch_rate_hz(broken, SFREQ) == pytest.approx(10.0)


@pytest.mark.unit
def test_pure_epoch_periodic_signal_is_all_residual_artifact():
    """A signal that is nothing but the epoch harmonic: comb RMS == signal RMS.

    Amplitude 10 µV sine at exactly the epoch rate has RMS 10/sqrt(2) µV, and
    every bit of its power sits on the first harmonic.
    """
    n = N_EPOCHS * PERIOD
    t = np.arange(n) / SFREQ
    sig = 10e-6 * np.sin(2 * np.pi * 10.0 * t)
    data = np.tile(sig, (N_CHANNELS, 1))

    ctx = GradientArtifactResidualCalculator(fmax=70.0).execute(_context(data, _triggers()))
    res = ctx.metadata.custom["gradient_artifact_residual"]

    assert res["epoch_rate_hz"] == pytest.approx(10.0)
    assert res["comb_share_pct"] == pytest.approx(100.0, abs=1.0)
    assert res["comb_rms_uv"] == pytest.approx(10.0 / np.sqrt(2), rel=0.02)
    assert res["n_harmonics"] == 6                  # 10, 20, 30, 40, 50, 60 Hz


@pytest.mark.unit
def test_off_harmonic_signal_is_not_counted_as_artifact():
    """A 7 Hz sine is not epoch-periodic, so almost none of it lands on the comb."""
    n = N_EPOCHS * PERIOD
    t = np.arange(n) / SFREQ
    data = np.tile(10e-6 * np.sin(2 * np.pi * 7.0 * t), (N_CHANNELS, 1))

    ctx = GradientArtifactResidualCalculator(fmax=70.0).execute(_context(data, _triggers()))
    res = ctx.metadata.custom["gradient_artifact_residual"]

    assert res["comb_share_pct"] < 1.0
    assert res["comb_rms_uv"] < 0.1 * res["rms_uv"]


@pytest.mark.unit
def test_comb_rms_is_amplitude_not_share():
    """The point of reporting µV: deleting the signal must not look like success.

    Both arms carry the *same* 1 µV epoch-periodic remnant; the second also keeps
    20 µV of broadband EEG. A share-based metric calls the first catastrophic
    (100 %) and the second clean (0.3 %); the µV figure says they left the same
    artifact, which is the true statement.

    The window is long deliberately. A 1 µV line under 20 µV of noise sits near
    the detection floor, and the estimate approaches the truth from above as the
    window grows — see the test below, which is where that property is pinned.
    """
    n_epochs = 960
    n = n_epochs * PERIOD
    t = np.arange(n) / SFREQ
    trg = np.arange(n_epochs) * PERIOD
    remnant = 1e-6 * np.sin(2 * np.pi * 10.0 * t)
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((N_CHANNELS, n)) * 20e-6

    deleted = GradientArtifactResidualCalculator().execute(
        _context(np.tile(remnant, (N_CHANNELS, 1)), trg)
    ).metadata.custom["gradient_artifact_residual"]
    kept = GradientArtifactResidualCalculator().execute(
        _context(eeg + remnant, trg)
    ).metadata.custom["gradient_artifact_residual"]

    assert deleted["comb_share_pct"] > 90.0
    assert kept["comb_share_pct"] < 5.0
    assert deleted["comb_rms_uv"] == pytest.approx(1.0 / np.sqrt(2), rel=0.05)
    assert kept["comb_rms_uv"] == pytest.approx(deleted["comb_rms_uv"], rel=0.2)


@pytest.mark.unit
def test_background_subtraction_converges_to_the_true_line_amplitude():
    """The broadband floor must shrink with the window, not sit there as a bias.

    Without subtracting the local background the estimate would stay proportional
    to the broadband level however long the recording; with it, the residual is
    finite-window noise and decreases monotonically towards the true 0.707 µV.
    """
    truth = 1.0 / np.sqrt(2)
    estimates = []
    for n_epochs in (60, 240, 960):
        n = n_epochs * PERIOD
        t = np.arange(n) / SFREQ
        rng = np.random.default_rng(0)
        data = rng.standard_normal((N_CHANNELS, n)) * 20e-6 + 1e-6 * np.sin(2 * np.pi * 10.0 * t)
        res = GradientArtifactResidualCalculator().execute(
            _context(data, np.arange(n_epochs) * PERIOD)
        ).metadata.custom["gradient_artifact_residual"]
        estimates.append(res["comb_rms_uv"])

    assert estimates == sorted(estimates, reverse=True), estimates
    assert all(e >= truth * 0.95 for e in estimates)
    assert estimates[-1] < estimates[0] * 0.85


@pytest.mark.unit
def test_background_subtraction_can_be_disabled():
    """``background_bins=0`` gives the raw comb power, floor included."""
    n = 240 * PERIOD
    rng = np.random.default_rng(3)
    data = rng.standard_normal((N_CHANNELS, n)) * 20e-6
    trg = np.arange(240) * PERIOD

    raw_comb = GradientArtifactResidualCalculator(background_bins=0.0).execute(
        _context(data, trg)).metadata.custom["gradient_artifact_residual"]
    subtracted = GradientArtifactResidualCalculator().execute(
        _context(data, trg)).metadata.custom["gradient_artifact_residual"]

    # Pure noise, no line at all: subtraction must remove most of the apparent comb.
    assert subtracted["comb_rms_uv"] < 0.5 * raw_comb["comb_rms_uv"]
    assert raw_comb["comb_share_pct"] == pytest.approx(raw_comb["comb_share_raw_pct"])


@pytest.mark.unit
def test_seam_step_is_one_for_a_continuous_signal():
    """No discontinuity at the joins -> the seam step is an ordinary sample step."""
    n = N_EPOCHS * PERIOD
    rng = np.random.default_rng(1)
    data = rng.standard_normal((N_CHANNELS, n)) * 10e-6

    ctx = EpochSeamStepCalculator().execute(_context(data, _triggers()))
    res = ctx.metadata.custom["epoch_seam_step"]

    assert res["n_seams"] == N_EPOCHS - 1
    assert res["ratio"] == pytest.approx(1.0, abs=0.35)


@pytest.mark.unit
def test_seam_step_detects_per_epoch_offsets():
    """Give every epoch its own baseline and the ratio must rise well above 1.

    This is the defect the metric exists for: per-segment demeaning, invisible to
    an evaluation that scores one epoch at a time.
    """
    n = N_EPOCHS * PERIOD
    rng = np.random.default_rng(2)
    data = rng.standard_normal((N_CHANNELS, n)) * 1e-6
    offsets = rng.standard_normal(N_EPOCHS) * 50e-6
    for e, off in enumerate(offsets):
        data[:, e * PERIOD:(e + 1) * PERIOD] += off

    ctx = EpochSeamStepCalculator().execute(_context(data, _triggers()))
    assert ctx.metadata.custom["epoch_seam_step"]["ratio"] > 10.0


@pytest.mark.unit
def test_seam_step_reports_nan_below_the_minimum():
    """Too few seams for a median: NaN, not a number nobody should trust."""
    n = 4 * PERIOD
    data = np.zeros((N_CHANNELS, n))
    ctx = EpochSeamStepCalculator().execute(_context(data, np.arange(4) * PERIOD))
    res = ctx.metadata.custom["epoch_seam_step"]
    assert np.isnan(res["ratio"])
    assert res["n_seams"] < EpochSeamStepCalculator.MIN_SEAMS


@pytest.mark.unit
def test_window_restriction_changes_what_is_measured():
    """The window argument must actually restrict the analysis.

    The first half is artifact-free, the second half carries the comb. Measuring
    the whole recording must land between the two halves.
    """
    n = N_EPOCHS * PERIOD
    t = np.arange(n) / SFREQ
    comb = 10e-6 * np.sin(2 * np.pi * 10.0 * t)
    comb[: n // 2] = 0.0
    data = np.tile(comb, (N_CHANNELS, 1))
    trg = _triggers()

    whole = GradientArtifactResidualCalculator().execute(
        _context(data, trg)).metadata.custom["gradient_artifact_residual"]
    second = GradientArtifactResidualCalculator(tmin=n / 2 / SFREQ).execute(
        _context(data, trg)).metadata.custom["gradient_artifact_residual"]

    assert second["comb_rms_uv"] > whole["comb_rms_uv"]
    assert second["comb_rms_uv"] == pytest.approx(10.0 / np.sqrt(2), rel=0.05)
