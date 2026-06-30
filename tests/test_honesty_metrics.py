"""Tests for the honesty metrics — does the model recover the *real* artifact?"""

from __future__ import annotations

import numpy as np

from facet.training import honesty_report, honesty_verdict


def _scenario(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    clean = rng.standard_normal(n)                      # true brain
    artifact = 30.0 * np.sin(np.linspace(0, 80 * np.pi, n))  # big structured artifact
    noisy = clean + artifact
    return noisy, clean, artifact


def test_perfect_recovery_is_honest():
    noisy, clean, _ = _scenario()
    report = honesty_report(noisy, output=clean, clean=clean)  # perfect output == clean
    assert report["artifact_corr"] > 0.999
    assert abs(report["residual_vs_clean_corr"]) < 1e-6
    assert honesty_verdict(report)["honest"] is True


def test_plausible_fabrication_is_caught():
    # Model outputs a *generic* EEG-like signal (independent of the true clean) —
    # "looks like EEG but isn't the true one". artifact_corr collapses, and the
    # true signal leaks into the residual.
    noisy, clean, _ = _scenario()
    fake = np.random.default_rng(99).standard_normal(clean.size)  # unrelated EEG-like
    report = honesty_report(noisy, output=fake, clean=clean)
    # clean_corr is the decisive discriminator: a generic fabrication does NOT
    # match the specific clean (~0), and the real signal leaks into the error.
    assert abs(report["clean_corr"]) < 0.3
    assert abs(report["residual_vs_clean_corr"]) > 0.5      # real signal sits in the error
    assert honesty_verdict(report)["honest"] is False


def test_under_subtraction_shows_artifact_in_residual():
    # Model leaves half the artifact in → residual correlates with the artifact.
    noisy, clean, artifact = _scenario()
    output = clean + 0.5 * artifact
    report = honesty_report(noisy, output=output, clean=clean)
    assert abs(report["residual_vs_artifact_corr"]) > 0.9
    assert honesty_verdict(report)["honest"] is False


def test_report_keys_present():
    noisy, clean, _ = _scenario()
    report = honesty_report(noisy, output=clean, clean=clean)
    for k in ("artifact_corr", "clean_corr", "residual_rms_ratio",
              "residual_vs_clean_corr", "residual_vs_artifact_corr", "artifact_rms_reduction"):
        assert k in report
