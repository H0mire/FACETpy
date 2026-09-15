"""Unit tests for the Run 6 spike-preservation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from facet.training.spike_metrics import compute_spike_metrics, expand_mask


def _case(n=4, t=400, spike_at=200, spike_amp=300e-6, seed=0):
    rng = np.random.default_rng(seed)
    clean = (rng.standard_normal((n, t)) * 5e-6).astype(np.float64)
    labels = np.zeros((n, t), dtype=bool)
    for i in range(n):
        clean[i, spike_at - 3:spike_at + 4] += spike_amp
        labels[i, spike_at - 3:spike_at + 4] = True
    return clean, labels


@pytest.mark.unit
def test_expand_mask_widens_symmetrically():
    m = np.zeros((1, 11), dtype=bool)
    m[0, 5] = True
    out = expand_mask(m, 2)
    assert out[0, 3:8].all()
    assert not out[0, :3].any() and not out[0, 8:].any()


@pytest.mark.unit
def test_perfect_reconstruction_is_ideal():
    clean, labels = _case()
    m = compute_spike_metrics(clean.copy(), clean, labels, neighborhood_samples=50)
    assert np.isinf(m["spike_neighborhood_snr_db"])      # zero error
    assert m["spike_amplitude_ratio"] == pytest.approx(1.0)
    assert m["spike_morphology_corr"] == pytest.approx(1.0)
    assert m["spike_peak_latency_drift_samples"] == pytest.approx(0.0)


@pytest.mark.unit
def test_residual_around_spike_lowers_neighbourhood_snr_and_contrast():
    """The discriminator: same spike, but one estimate leaves residual around it."""
    clean, labels = _case()
    rng = np.random.default_rng(1)
    residual = rng.standard_normal(clean.shape) * 40e-6      # leftover artifact
    dirty = clean + residual

    good = compute_spike_metrics(clean.copy(), clean, labels, neighborhood_samples=50)
    bad = compute_spike_metrics(dirty, clean, labels, neighborhood_samples=50)

    assert bad["spike_neighborhood_snr_db"] < good["spike_neighborhood_snr_db"]
    assert bad["spike_contrast_db"] < good["spike_contrast_db"]
    # ...while the spike itself is essentially untouched in both
    assert bad["spike_amplitude_ratio"] == pytest.approx(1.0, abs=0.3)


@pytest.mark.unit
def test_attenuated_spike_shows_up_in_amplitude_ratio():
    """The DL-specific failure: learning the spike away."""
    clean, labels = _case()
    flattened = clean.copy()
    flattened[labels] *= 0.4
    m = compute_spike_metrics(flattened, clean, labels, neighborhood_samples=50)
    assert m["spike_amplitude_ratio"] < 0.6


@pytest.mark.unit
def test_shape_mismatch_raises():
    clean, labels = _case()
    with pytest.raises(ValueError, match="shapes must match"):
        compute_spike_metrics(clean[:, :-1], clean, labels)


@pytest.mark.unit
def test_no_spikes_yields_nan_spike_metrics_but_valid_bulk():
    rng = np.random.default_rng(2)
    clean = rng.standard_normal((3, 200)) * 5e-6
    labels = np.zeros((3, 200), dtype=bool)
    m = compute_spike_metrics(clean + 1e-6, clean, labels, neighborhood_samples=10)
    assert np.isnan(m["spike_amplitude_ratio"])
    assert m["n_spike_examples"] == 0.0
    assert np.isfinite(m["non_spike_snr_db"])


# ---------------------------------------------------------------------------
# Spike-aware loss (run_6 Phase C)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


@pytest.mark.unit
def test_spike_weighted_loss_reduces_to_mse_when_no_spike():
    from facet.training.weg_a_baseline import SpikeWeightedMSELoss

    loss = SpikeWeightedMSELoss(spike_weight=20.0)
    pred = torch.randn(3, 1, 16)
    target = torch.zeros(3, 2, 16)
    target[:, 0] = torch.randn(3, 16)          # artifact, no spike anywhere
    expected = torch.nn.functional.mse_loss(pred, target[:, :1])
    assert float(loss(pred, target)) == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.unit
def test_spike_weighted_loss_penalises_spike_region_harder():
    from facet.training.weg_a_baseline import SpikeWeightedMSELoss

    loss = SpikeWeightedMSELoss(spike_weight=10.0)
    target = torch.zeros(1, 2, 10)
    target[0, 1, 4:6] = 1.0                    # spike on 2 of 10 samples
    err_in = torch.zeros(1, 1, 10)
    err_in[0, 0, 4:6] = 1.0                    # error only on the spike
    err_out = torch.zeros(1, 1, 10)
    err_out[0, 0, 0:2] = 1.0                   # same-size error elsewhere
    assert float(loss(err_in, target)) > float(loss(err_out, target))


@pytest.mark.unit
def test_spike_weighted_loss_requires_the_mask():
    from facet.training.weg_a_baseline import SpikeWeightedMSELoss

    loss = SpikeWeightedMSELoss()
    with pytest.raises(ValueError, match="spike mask"):
        loss(torch.zeros(1, 1, 8), torch.zeros(1, 1, 8))


# ---------------------------------------------------------------------------
# Cascade / residual formulation (run_6 §3c)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_residual_mode_feeds_farm_corrected_input_and_residual_target(tmp_path):
    """The reformulation: the model must see what FARM leaves, not the raw artifact.

    Predicting the whole artifact is a ~40 dB problem (1961 uV over 19 uV) that no
    loss fixed. In residual mode the input is noisy - template and the target is
    artifact - template, which is ~6:1 instead.
    """
    import numpy as np

    from facet.training import NPZSpatioTemporalDataset

    n, ep, ch, core, guard = 4, 3, 2, 16, 4
    length = core + 2 * guard
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((n, ep, ch, length)).astype(np.float32) * 1e-6
    template = rng.standard_normal((n, ep, ch, length)).astype(np.float32) * 100e-6
    residual = rng.standard_normal((n, ep, ch, length)).astype(np.float32) * 5e-6
    artifact = template + residual
    path = tmp_path / "resid.npz"
    np.savez_compressed(
        path,
        clean_context=clean, artifact_context=artifact,
        artifact_context_template=template,
        clean_center=clean[:, ep // 2, 0:1],
        artifact_center=artifact[:, ep // 2, 0:1],
        artifact_center_template=template[:, ep // 2, 0:1],
        spike_labels=np.zeros((n, 1, length), np.float32),
        neighbor_channel_indices=np.zeros((n, ch), np.int64),
        target_channel_index=np.zeros(n, np.int64),
        center_epoch_index=np.arange(n, dtype=np.int64),
        example_split=np.array([0, 0, 1, 1], dtype=np.int64),
        core_samples=np.asarray([core]), guard_samples=np.asarray([guard]),
        context_epochs=np.asarray([ep]), k_neighbors=np.asarray([ch - 1]),
        sfreq=np.asarray([1000.0]), ch_names=np.asarray(["C3", "C4"], dtype=object),
        clean_source=np.asarray(["synthetic"], dtype=object),
        spikes_injected=np.asarray([False]), n_examples=np.asarray([n]),
    )

    plain = NPZSpatioTemporalDataset(path, max_shift=0)
    resid = NPZSpatioTemporalDataset(path, max_shift=0, residual_mode=True)
    noisy_plain, target_plain = plain[0]
    noisy_resid, target_resid = resid[0]

    # the residual input is far smaller than the raw one — that is the whole point
    assert np.abs(noisy_resid).mean() < 0.2 * np.abs(noisy_plain).mean()
    # and the target is the residual, not the full artifact
    expected = (artifact - template)[0, ep // 2, 0:1, guard:guard + core]
    np.testing.assert_allclose(target_resid, expected, rtol=1e-4, atol=1e-12)
    assert np.abs(target_resid).mean() < 0.2 * np.abs(target_plain).mean()


@pytest.mark.unit
def test_residual_mode_requires_the_template(tmp_path):
    import numpy as np

    from facet.training import NPZSpatioTemporalDataset

    n, ep, ch, core, guard = 2, 3, 2, 8, 2
    length = core + 2 * guard
    path = tmp_path / "no_template.npz"
    np.savez_compressed(
        path,
        clean_context=np.zeros((n, ep, ch, length), np.float32),
        artifact_context=np.zeros((n, ep, ch, length), np.float32),
        clean_center=np.zeros((n, 1, length), np.float32),
        artifact_center=np.zeros((n, 1, length), np.float32),
        spike_labels=np.zeros((n, 1, length), np.float32),
        neighbor_channel_indices=np.zeros((n, ch), np.int64),
        target_channel_index=np.zeros(n, np.int64),
        center_epoch_index=np.arange(n, dtype=np.int64),
        core_samples=np.asarray([core]), guard_samples=np.asarray([guard]),
        context_epochs=np.asarray([ep]), k_neighbors=np.asarray([ch - 1]),
        sfreq=np.asarray([1000.0]), ch_names=np.asarray(["C3", "C4"], dtype=object),
        clean_source=np.asarray(["synthetic"], dtype=object),
        spikes_injected=np.asarray([False]), n_examples=np.asarray([n]),
    )
    with pytest.raises(ValueError, match="residual_mode requires"):
        NPZSpatioTemporalDataset(path, residual_mode=True)


# ---------------------------------------------------------------------------
# Spatial-context ablation (run_7)
# ---------------------------------------------------------------------------


def _write_spatiotemporal_npz(path, n=4, ep=3, ch=4, core=16, guard=4):
    import numpy as np

    length = core + 2 * guard
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((n, ep, ch, length)).astype(np.float32)
    artifact = rng.standard_normal((n, ep, ch, length)).astype(np.float32)
    np.savez_compressed(
        path,
        clean_context=clean,
        artifact_context=artifact,
        artifact_context_template=artifact * 0.9,
        clean_center=clean[:, ep // 2, 0:1],
        artifact_center=artifact[:, ep // 2, 0:1],
        artifact_center_template=artifact[:, ep // 2, 0:1] * 0.9,
        spike_labels=np.zeros((n, 1, length), np.float32),
        neighbor_channel_indices=np.tile(np.arange(ch), (n, 1)).astype(np.int64),
        target_channel_index=np.zeros(n, np.int64),
        center_epoch_index=np.arange(n, dtype=np.int64),
        example_split=np.array([0, 0, 1, 1], dtype=np.int64),
        core_samples=np.asarray([core]),
        guard_samples=np.asarray([guard]),
        context_epochs=np.asarray([ep]),
        k_neighbors=np.asarray([ch - 1]),
        sfreq=np.asarray([1000.0]),
        ch_names=np.asarray(["C3", "C4", "Cz", "Pz"], dtype=object),
        clean_source=np.asarray(["synthetic"], dtype=object),
        spikes_injected=np.asarray([False]),
        n_examples=np.asarray([n]),
    )
    return clean, artifact


@pytest.mark.unit
def test_max_channels_keeps_the_target_channel_first(tmp_path):
    """Ablating spatial context must drop neighbours, never the target electrode.

    The builder writes the target channel at index 0 and its geodesic neighbours
    after it, so a prefix is 'target plus nearest N-1'. Dropping from the front
    instead would silently train every model to reconstruct a neighbour.
    """
    import numpy as np

    from facet.training import NPZSpatioTemporalDataset

    path = tmp_path / "ctx.npz"
    clean, artifact = _write_spatiotemporal_npz(path)

    full = NPZSpatioTemporalDataset(path, max_shift=0)
    single = NPZSpatioTemporalDataset(path, max_shift=0, max_channels=1)
    assert full.input_shape[1] == 4
    assert single.input_shape[1] == 1

    noisy_full, target_full = full[0]
    noisy_single, target_single = single[0]
    np.testing.assert_allclose(noisy_single[:, 0], noisy_full[:, 0], rtol=1e-6)
    np.testing.assert_allclose(target_single, target_full, rtol=1e-6)


@pytest.mark.unit
def test_max_channels_is_clamped_to_the_available_channels(tmp_path):
    from facet.training import NPZSpatioTemporalDataset

    path = tmp_path / "ctx.npz"
    _write_spatiotemporal_npz(path)
    assert NPZSpatioTemporalDataset(path, max_shift=0, max_channels=99).input_shape[1] == 4
    assert NPZSpatioTemporalDataset(path, max_shift=0, max_channels=0).input_shape[1] == 1


@pytest.mark.unit
def test_max_examples_keeps_both_stored_splits(tmp_path):
    """A subsampled run must still have a validation set.

    The builder stores validation as a contiguous tail, so a plain ``[:limit]``
    kept only training examples and the trainer failed with an unpack error far
    from the cause.
    """
    from facet.training import NPZSpatioTemporalDataset

    path = tmp_path / "ctx.npz"
    _write_spatiotemporal_npz(path)
    subset = NPZSpatioTemporalDataset(path, max_shift=0, max_examples=2)
    train, val = subset.train_val_split(val_ratio=0.2, seed=0)
    assert len(train) > 0 and len(val) > 0
