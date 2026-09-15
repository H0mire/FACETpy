"""Closed-form tests for the recovered-clean objective.

The claim the module is built on is a numeric one — "a deleted signal scores
exactly 1.0 per time-domain term, a perfect one exactly 0.0" — so it is tested as
a numeric one. Without that anchor the normalisation is just a division and the
next refactor is free to change what the loss means.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facet.training.deployment_losses import (  # noqa: E402
    RecoveredCleanObjective,
    build_deployment_loss,
)

ROWS = ("artifact", "clean", "noisy")
B, C, S = 4, 3, 256
SFREQ = 1000.0


def _batch(seed: int = 0):
    """A target stack whose rows satisfy ``noisy = clean + artifact`` exactly."""
    rng = np.random.default_rng(seed)
    clean = rng.standard_normal((B, C, S)).astype(np.float32)
    t = np.arange(S, dtype=np.float32) / SFREQ
    artifact = (20.0 * np.sin(2 * np.pi * 7.0 * t)).astype(np.float32)[None, None, :]
    artifact = np.broadcast_to(artifact, (B, C, S)).copy()
    noisy = clean + artifact
    stack = np.stack([artifact, clean, noisy], axis=1)  # (B, 3, C, S)
    return (torch.from_numpy(artifact), torch.from_numpy(clean), torch.from_numpy(noisy), torch.from_numpy(stack))


@pytest.mark.unit
def test_perfect_artifact_prediction_scores_zero():
    artifact, _, _, target = _batch()
    loss = RecoveredCleanObjective(prediction_is="artifact", rows=ROWS, si_sdr_weight=0.0, sfreq=SFREQ)
    assert float(loss(artifact, target)) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.unit
def test_deleting_the_signal_scores_one_per_time_domain_term():
    """``prediction = noisy`` means ``clean_hat = 0``. Each term must read 1.0.

    This is the failure mode the module exists for, and 1.0 is not an
    approximation: normalised MSE of a zero estimate is mean(ref^2)/mean(ref^2).
    """
    _, _, noisy, target = _batch()
    loss = RecoveredCleanObjective(prediction_is="artifact", rows=ROWS, si_sdr_weight=0.0, sfreq=SFREQ)
    total = float(loss(noisy, target))

    for term in ("amplitude", "velocity", "acceleration", "frequency"):
        assert loss.last_terms[term] == pytest.approx(1.0, rel=1e-4), term
    assert total == pytest.approx(4.0, rel=1e-4)
    assert loss.last_terms["energy_ratio"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_doing_nothing_is_also_penalised():
    """``prediction = 0`` leaves the whole artifact in. Not the deletion failure,
    but it must not be cheap either — the artifact is far bigger than the EEG."""
    _, _, _, target = _batch()
    loss = RecoveredCleanObjective(prediction_is="artifact", rows=ROWS, si_sdr_weight=0.0, sfreq=SFREQ)
    total = float(loss(torch.zeros(B, C, S), target))
    assert total > 4.0
    assert loss.last_terms["energy_ratio"] > 1.0


@pytest.mark.unit
def test_a_constant_output_is_caught_by_the_derivative_terms_alone():
    """Amplitude-only cannot distinguish a per-epoch offset; velocity can.

    ``clean_hat`` here is the true clean plus a per-example constant. The
    amplitude term sees the offset, but the derivative terms see it as *zero*
    error — which is the point of the pairing: run the same test with a flat
    ``clean_hat`` and the derivatives are what fire.
    """
    artifact, _, noisy, target = _batch()
    offset = torch.tensor([0.0, 5.0, -5.0, 10.0]).reshape(B, 1, 1)
    loss = RecoveredCleanObjective(prediction_is="artifact", rows=ROWS, si_sdr_weight=0.0, sfreq=SFREQ)

    loss(artifact - offset, target)  # clean_hat = clean + offset
    assert loss.last_terms["amplitude"] > 0.5
    assert loss.last_terms["velocity"] == pytest.approx(0.0, abs=1e-5)

    loss(noisy - offset, target)  # clean_hat = offset, a flat line
    assert loss.last_terms["velocity"] == pytest.approx(1.0, rel=1e-4)
    assert loss.last_terms["acceleration"] == pytest.approx(1.0, rel=1e-4)
    assert loss.last_terms["energy_ratio"] > 0.0  # not zero: a non-zero constant


@pytest.mark.unit
def test_si_sdr_term_rewards_a_good_fit_and_not_a_deleted_one():
    artifact, _, noisy, target = _batch()
    loss = RecoveredCleanObjective(
        prediction_is="artifact",
        rows=ROWS,
        amplitude_weight=0.0,
        velocity_weight=0.0,
        acceleration_weight=0.0,
        frequency_weight=0.0,
        si_sdr_weight=1.0,
        si_sdr_max=30.0,
        sfreq=SFREQ,
    )
    good = float(loss(artifact, target))
    deleted = float(loss(noisy, target))
    assert good == pytest.approx(-1.0, abs=1e-3)  # clamped at si_sdr_max
    assert deleted > good
    # Documented limitation: with the eps guard SI-SDR of an all-zero estimate is
    # 0 dB, not -inf. On its own it therefore only *ranks* deletion below a good
    # fit; the normalised time-domain terms are what put a floor under it.
    assert deleted == pytest.approx(0.0, abs=1e-3)


@pytest.mark.unit
def test_clean_prediction_mode_needs_no_noisy_row():
    _, clean, _, target = _batch()
    loss = RecoveredCleanObjective(prediction_is="clean", rows=ROWS, si_sdr_weight=0.0, sfreq=SFREQ)
    assert float(loss(clean, target)) == pytest.approx(0.0, abs=1e-5)
    loss(torch.zeros(B, C, S), target)
    assert loss.last_terms["amplitude"] == pytest.approx(1.0, rel=1e-4)


@pytest.mark.unit
def test_row_layout_is_checked_rather_than_assumed():
    _, clean, _, target = _batch()
    with pytest.raises(ValueError, match="must contain 'clean'"):
        RecoveredCleanObjective(prediction_is="clean", rows=("artifact",))
    with pytest.raises(ValueError, match="must contain 'noisy'"):
        RecoveredCleanObjective(prediction_is="artifact", rows=("artifact", "clean"))
    loss = RecoveredCleanObjective(prediction_is="clean", rows=("clean",), sfreq=SFREQ)
    with pytest.raises(ValueError, match="rows on axis 1"):
        loss(clean, target)


@pytest.mark.unit
def test_identity_term_is_scale_free_and_one_at_the_truth():
    artifact, _, _, target = _batch()
    loss = RecoveredCleanObjective(
        prediction_is="artifact",
        rows=ROWS,
        amplitude_weight=0.0,
        velocity_weight=0.0,
        acceleration_weight=0.0,
        frequency_weight=0.0,
        si_sdr_weight=0.0,
        identity_weight=1.0,
        sfreq=SFREQ,
    )
    assert float(loss(artifact, target)) == pytest.approx(1.0, rel=1e-5)
    assert float(loss(2.0 * artifact, target)) == pytest.approx(2.0, rel=1e-5)

    hinged = RecoveredCleanObjective(
        prediction_is="artifact",
        rows=ROWS,
        amplitude_weight=0.0,
        velocity_weight=0.0,
        acceleration_weight=0.0,
        frequency_weight=0.0,
        si_sdr_weight=0.0,
        identity_weight=1.0,
        identity_hinge=True,
        sfreq=SFREQ,
    )
    assert float(hinged(artifact, target)) == pytest.approx(0.0, abs=1e-5)
    assert float(hinged(2.0 * artifact, target)) == pytest.approx(1.0, rel=1e-5)


@pytest.mark.unit
def test_frequency_band_actually_restricts_the_term():
    """The 7 Hz artifact sits inside 1-70 Hz and outside 100-200 Hz."""
    artifact, _, noisy, target = _batch()
    kw = dict(
        prediction_is="artifact",
        rows=ROWS,
        amplitude_weight=0.0,
        velocity_weight=0.0,
        acceleration_weight=0.0,
        frequency_weight=1.0,
        si_sdr_weight=0.0,
        sfreq=SFREQ,
    )
    in_band = float(RecoveredCleanObjective(freq_band=(1.0, 70.0), **kw)(noisy, target))
    out_band = float(RecoveredCleanObjective(freq_band=(100.0, 200.0), **kw)(noisy, target))
    assert in_band == pytest.approx(1.0, rel=1e-3)
    assert out_band == pytest.approx(1.0, rel=1e-3)  # deletion is total either way
    # But a prediction that is only wrong at 7 Hz must be invisible out of band.
    partial = artifact * 0.5
    assert float(RecoveredCleanObjective(freq_band=(100.0, 200.0), **kw)(partial, target)) < float(
        RecoveredCleanObjective(freq_band=(1.0, 70.0), **kw)(partial, target)
    )


@pytest.mark.unit
def test_ensemble_factory_drops_the_scale_invariant_term():
    loss = build_deployment_loss("ic_unet_ensemble", rows=ROWS, sfreq=SFREQ)
    assert loss.si_sdr_weight == 0.0
    assert (loss.amplitude_weight, loss.velocity_weight, loss.acceleration_weight, loss.frequency_weight) == (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    with pytest.raises(ValueError, match="Unknown deployment loss"):
        build_deployment_loss("nope")


# --------------------------------------------------------------------------
# Scale-free early stopping
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_relative_min_delta_follows_the_loss_scale():
    """The bug this fixes: an absolute threshold larger than the whole loss.

    With ``min_delta=1e-6`` a loss around 2.4e-07 -- the ``vit_spectrogram``
    run's actual scale -- can never improve, so training stops after
    ``patience`` epochs no matter what the model does. A relative threshold
    asks for a 1 % improvement whatever the units are.
    """
    from facet.training.callbacks import EarlyStoppingCallback

    absolute = EarlyStoppingCallback(monitor="val_loss", min_delta=1e-6)
    assert not absolute._is_better(2.0e-07, 2.45e-07)  # a 18 % gain, rejected

    relative = EarlyStoppingCallback(monitor="val_loss", min_delta=0.0, min_delta_rel=0.01)
    assert relative._is_better(2.0e-07, 2.45e-07)  # same gain, accepted
    assert not relative._is_better(2.44e-07, 2.45e-07)  # 0.4 %, still rejected
    assert relative._is_better(0.90, 1.00)  # and scale-free
    assert not relative._is_better(0.999, 1.00)


@pytest.mark.unit
def test_relative_min_delta_defaults_to_the_old_behaviour():
    from facet.training.callbacks import EarlyStoppingCallback

    cb = EarlyStoppingCallback(monitor="val_loss", min_delta=1e-4)
    assert cb.min_delta_rel == 0.0
    assert cb._threshold(1.0) == pytest.approx(1e-4)
    assert cb._threshold(float("inf")) == pytest.approx(1e-4)
    assert cb._is_better(0.5, 1.0) and not cb._is_better(1.0, 1.0)


@pytest.mark.unit
def test_context_dataset_extras_are_opt_in_and_aligned():
    """The extra rows must decompose exactly, or ``noisy - prediction`` is wrong."""
    from pathlib import Path

    from facet.training.dataset import NPZContextArtifactDataset

    path = Path("output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz")
    if not path.exists():
        pytest.skip("proof-fit bundle not built in this checkout")

    plain = NPZContextArtifactDataset(path, max_examples=4)
    assert plain[0][1].ndim == 2 and plain.target_rows == ("artifact",)

    stacked = NPZContextArtifactDataset(path, max_examples=4, target_extras=("clean", "noisy"))
    _, target = stacked[0]
    assert stacked.target_rows == ("artifact", "clean", "noisy")
    assert target.shape == (3,) + plain[0][1].shape
    assert np.abs(target[2] - (target[0] + target[1])).max() == 0.0


@pytest.mark.unit
def test_the_loss_does_not_depend_on_the_unit_the_data_is_stored_in():
    """Volts or microvolts must give the same number.

    They did not: every denominator carried an absolute ``+ 1e-8``, which is 4 %
    of the mean square of an EEG signal stored in volts and negligible for the
    same signal in microvolts. "Delete the whole signal" scored 0.976 in one unit
    and 1.000 in the other, and the whole design rests on that value being 1.0.
    """
    _, _, noisy, target = _batch()
    values = []
    for scale in (1.0, 1e-6, 1e6):
        loss = RecoveredCleanObjective(prediction_is="artifact", rows=ROWS, sfreq=SFREQ)
        loss(noisy * scale, target * scale)
        values.append(dict(loss.last_terms))

    for term in ("amplitude", "velocity", "acceleration", "frequency"):
        assert values[0][term] == pytest.approx(1.0, rel=1e-5), term
        for other in values[1:]:
            assert other[term] == pytest.approx(values[0][term], rel=1e-4), term


@pytest.mark.unit
def test_si_sdr_is_also_unit_free():
    artifact, _, _, target = _batch()
    out = []
    for scale in (1.0, 1e-6):
        loss = RecoveredCleanObjective(
            prediction_is="artifact",
            rows=ROWS,
            amplitude_weight=0.0,
            velocity_weight=0.0,
            acceleration_weight=0.0,
            frequency_weight=0.0,
            si_sdr_weight=1.0,
            sfreq=SFREQ,
        )
        out.append(float(loss(artifact * scale * 1.001, target * scale)))
    assert out[0] == pytest.approx(out[1], rel=1e-4)
