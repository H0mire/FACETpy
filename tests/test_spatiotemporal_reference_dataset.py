"""Cheap CPU smoke tests for the Run 3 / Weg A spatio-temporal reference dataset.

Covers the builder (shapes + guard band), the four augmentation transforms as
pure functions (shift / mix / amplitude / length invariants), the dataset
contract, and the spike-injection mode. Uses a tiny synthetic bundle so it runs
in well under a second on an M4 Pro — no real recording or GPU needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.training import (
    AmplitudeJitter,
    BackgroundMix,
    LengthJitterNoise,
    NPZSpatioTemporalDataset,
    WindowShift,
)
from facet.training.spatiotemporal_builder import (
    build_spatiotemporal_reference_dataset,
    resample_and_tile,
    select_neighbors,
)

# Real montage names so the geodesic k-NN path is exercised.
CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]


def _toy_bundle(n_channels: int = 8, n_epochs: int = 12, epoch_len: int = 40) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    sfreq = 1000.0
    n_samples = n_epochs * epoch_len
    triggers = np.arange(0, n_samples, epoch_len, dtype=np.int64)
    # Artifact: a per-channel periodic template shared across epochs (the thing
    # that is correlated over epochs). Corrected: independent per-channel noise.
    t = np.arange(epoch_len)
    template = np.stack([np.sin(2 * np.pi * (c + 1) * t / epoch_len) for c in range(n_channels)])
    artifact = np.tile(template, (1, n_epochs)).astype(np.float32) * 5.0
    corrected = (rng.standard_normal((n_channels, n_samples)) * 1.0).astype(np.float32)
    return {
        "artifact": artifact,
        "corrected": corrected,
        "triggers": triggers,
        "sfreq": np.asarray([sfreq], dtype=np.float64),
        "artifact_to_trigger_offset": np.asarray([0.0], dtype=np.float64),
        "ch_names": np.asarray(CH_NAMES[:n_channels], dtype=object),
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_select_neighbors_returns_self_plus_k():
    neigh = select_neighbors(CH_NAMES, k_neighbors=2)
    assert neigh.shape == (len(CH_NAMES), 3)
    # self is always first (geodesic distance 0)
    assert np.array_equal(neigh[:, 0], np.arange(len(CH_NAMES)))
    # no channel repeats within a triple
    for row in neigh:
        assert len(set(row.tolist())) == 3


def test_builder_shapes_and_guard_band():
    bundle = _toy_bundle()
    core, guard, ctx = 32, 8, 7
    ds = build_spatiotemporal_reference_dataset(
        bundle, context_epochs=ctx, core_samples=core, guard_samples=guard,
        k_neighbors=2, clean_source="aas_corrected", seed=0,
    )
    out_len = core + 2 * guard
    n = int(ds["n_examples"][0])
    # noisy_context is derivable (clean + artifact) and intentionally not stored
    assert "noisy_context" not in ds
    assert ds["clean_context"].shape == (n, ctx, 3, out_len)
    assert ds["artifact_context"].shape == (n, ctx, 3, out_len)
    assert ds["artifact_center"].shape == (n, 1, out_len)
    assert ds["clean_center"].shape == (n, 1, out_len)
    # target is the artifact of the target channel (idx 0) at the center epoch
    np.testing.assert_allclose(ds["artifact_center"], ds["artifact_context"][:, ctx // 2, 0:1])


def test_builder_synthetic_clean_is_independent_of_corrected():
    bundle = _toy_bundle()
    ds = build_spatiotemporal_reference_dataset(
        bundle, context_epochs=7, core_samples=32, guard_samples=8,
        clean_source="synthetic", seed=0,
    )
    assert str(ds["clean_source"][0]) == "synthetic"
    # synthetic clean is finite and not the all-zero placeholder
    assert np.isfinite(ds["clean_context"]).all()
    assert float(np.mean(np.abs(ds["clean_context"]))) > 0.0


def test_resample_and_tile_shape_and_rate():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((4, 1000)).astype(np.float32)  # 4 ch, 1000 samp @ 500 Hz
    out = resample_and_tile(clean, src_sfreq=500.0, dst_sfreq=1000.0, n_samples=5000)
    assert out.shape == (4, 5000)  # upsampled (1000->2000) then tiled to 5000
    assert np.isfinite(out).all()


def test_builder_niazy_pretrigger_uses_real_clean():
    bundle = _toy_bundle()  # 8 ch artifact bundle @ 1000 Hz
    n_ch = bundle["artifact"].shape[0]
    rng = np.random.default_rng(1)
    pretrigger = (rng.standard_normal((n_ch, 600)) * 2.0).astype(np.float32)  # short real-ish clean @ 500 Hz
    ds = build_spatiotemporal_reference_dataset(
        bundle, context_epochs=7, core_samples=32, guard_samples=8,
        clean_source="niazy_pretrigger", pretrigger_clean=pretrigger, pretrigger_sfreq=500.0, seed=0,
    )
    assert str(ds["clean_source"][0]) == "niazy_pretrigger"
    assert np.isfinite(ds["clean_context"]).all()
    assert float(np.mean(np.abs(ds["clean_context"]))) > 0.0


def test_builder_niazy_pretrigger_rejects_channel_mismatch():
    bundle = _toy_bundle()
    bad = np.zeros((3, 600), dtype=np.float32)  # wrong channel count
    with pytest.raises(ValueError, match="channels"):
        build_spatiotemporal_reference_dataset(
            bundle, context_epochs=7, core_samples=32, guard_samples=8,
            clean_source="niazy_pretrigger", pretrigger_clean=bad, pretrigger_sfreq=500.0,
        )


def test_builder_spike_mode_emits_labels():
    bundle = _toy_bundle(n_epochs=16)
    ds = build_spatiotemporal_reference_dataset(
        bundle, context_epochs=7, core_samples=64, guard_samples=8,
        clean_source="synthetic", inject_spikes_mode=True,
        spike_rate_hz=5.0, spike_amplitude_uv=80.0, seed=0,
    )
    assert bool(ds["spikes_injected"][0]) is True
    assert ds["spike_labels"].shape == ds["artifact_center"].shape
    assert float(np.sum(ds["spike_labels"])) > 0.0  # at least one labelled spike


# ---------------------------------------------------------------------------
# Transforms (pure)
# ---------------------------------------------------------------------------


def _toy_sample(core: int = 32, guard: int = 8, ch: int = 3, ep: int = 7) -> dict:
    out_len = core + 2 * guard
    rng = np.random.default_rng(1)
    artifact = rng.standard_normal((ep, ch, out_len)).astype(np.float32)
    clean = rng.standard_normal((ep, ch, out_len)).astype(np.float32)
    return {
        "noisy": (clean + artifact).astype(np.float32),
        "clean": clean,
        "artifact": artifact,
        "target": artifact[ep // 2, 0:1].copy(),
        "spike": np.zeros((1, out_len), np.float32),
    }


def test_window_shift_zero_is_exact_center_crop():
    core, guard = 32, 8
    sample = _toy_sample(core, guard)
    out = WindowShift(core_samples=core, max_shift=0)(sample)
    assert out["noisy"].shape == (7, 3, core)
    # δ=0 must equal the exact center slice [guard:guard+core], no interpolation
    np.testing.assert_array_equal(out["noisy"], sample["noisy"][..., guard:guard + core])


def test_window_shift_integer_offset_matches_slice():
    core, guard = 32, 8
    sample = _toy_sample(core, guard)
    shifter = WindowShift(core_samples=core, max_shift=4, seed=3)
    out = shifter(sample)
    # the cropped window must be an exact contiguous slice of the padded input
    noisy = sample["noisy"]
    found = None
    for start in range(0, noisy.shape[-1] - core + 1):
        if np.array_equal(out["noisy"], noisy[..., start:start + core]):
            found = start
            break
    assert found is not None, "integer WindowShift must yield an exact contiguous crop"


def test_background_mix_swaps_clean_keeps_artifact():
    sample = _toy_sample()
    alt = np.full_like(sample["clean"], 7.0)
    sample["clean_alt"] = alt
    out = BackgroundMix(prob=1.0)(sample)
    np.testing.assert_array_equal(out["clean"], alt)
    np.testing.assert_array_equal(out["artifact"], sample["artifact"])  # artifact untouched
    np.testing.assert_allclose(out["noisy"], alt + sample["artifact"], rtol=1e-5)


def test_amplitude_jitter_scales_consistently():
    sample = _toy_sample()
    out = AmplitudeJitter(global_range=(2.0, 2.0), per_channel_range=(1.0, 1.0))(sample)
    # constant ranges => exact ×2 on every signal, noisy=clean+artifact preserved
    np.testing.assert_allclose(out["noisy"], 2.0 * sample["noisy"], rtol=1e-5)
    np.testing.assert_allclose(out["noisy"], out["clean"] + out["artifact"], rtol=1e-4, atol=1e-4)


def test_length_jitter_noise_only_perturbs_noisy():
    sample = _toy_sample()
    out = LengthJitterNoise(length_eps=0.0, noise_std_frac=0.1, seed=0)(sample)
    assert out["noisy"].shape == sample["noisy"].shape
    # measurement noise lands on noisy only — artifact/target stay clean
    np.testing.assert_array_equal(out["artifact"], sample["artifact"])
    np.testing.assert_array_equal(out["target"], sample["target"])
    assert not np.array_equal(out["noisy"], sample["noisy"])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _write_dataset(tmp_path, **kwargs) -> str:
    bundle = _toy_bundle(n_epochs=14)
    ds = build_spatiotemporal_reference_dataset(
        bundle, context_epochs=7, core_samples=32, guard_samples=8, **kwargs
    )
    path = tmp_path / "spatiotemporal.npz"
    np.savez_compressed(path, **ds)
    return str(path)


def test_dataset_contract_and_shapes(tmp_path):
    path = _write_dataset(tmp_path, clean_source="aas_corrected", seed=0)
    ds = NPZSpatioTemporalDataset(path, max_shift=4, seed=0)
    assert len(ds) > 0
    assert ds.input_shape == (7, 3, 32)
    assert ds.target_shape == (1, 32)
    noisy, target = ds[0]
    assert noisy.shape == (7, 3, 32)
    assert target.shape == (1, 32)
    assert noisy.dtype == np.float32
    train, val = ds.train_val_split(val_ratio=0.25, seed=1)
    assert len(train) + len(val) == len(ds)
    n0, t0 = train[0]
    assert n0.shape == (7, 3, 32)


def test_dataset_background_mix_runs(tmp_path):
    path = _write_dataset(tmp_path, clean_source="aas_corrected", seed=0)
    ds = NPZSpatioTemporalDataset(path, max_shift=2, background_mix_prob=1.0, seed=0)
    noisy, target = ds[0]
    assert noisy.shape == (7, 3, 32)
    assert np.isfinite(noisy).all()


def test_dataset_clean_target_disables_background_mix(tmp_path):
    path = _write_dataset(tmp_path, clean_source="synthetic", seed=0)
    ds = NPZSpatioTemporalDataset(path, target_key="clean_center", background_mix_prob=1.0)
    assert ds.background_mix_prob == 0.0
    assert ds.target_type == "clean"


def test_dataset_torch_adapter(tmp_path):
    torch = pytest.importorskip("torch")
    path = _write_dataset(tmp_path, clean_source="aas_corrected", seed=0)
    ds = NPZSpatioTemporalDataset(path, seed=0)
    loader = torch.utils.data.DataLoader(ds.to_torch(), batch_size=2)
    noisy, target = next(iter(loader))
    assert noisy.shape == (2, 7, 3, 32)
    assert target.shape == (2, 1, 32)
