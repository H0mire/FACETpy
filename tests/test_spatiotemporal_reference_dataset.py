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
    inject_real_ieds,
    resample_and_tile,
    select_neighbors,
)


def _fake_ied_pool(names=("C3", "C4", "F3"), t=100, focal_uv=300.0):
    """A pool entry shaped like the real extractor's output.

    Volts (NOT focal-normalised), plus the ``marker`` offset of the '!' onset —
    matching ``extract_real_ied_pool`` after run_3 §6.6.
    """
    x = np.linspace(-3, 3, t)
    wave = (-x * np.exp(-(x**2))).astype(np.float64)   # sharp biphasic
    wave = wave / np.max(np.abs(wave)) * (focal_uv * 1e-6)
    waveforms = np.stack([wave, 0.5 * wave, 0.3 * wave]).astype(np.float32)
    return [{"waveforms": waveforms, "names": list(names), "marker": t // 2}]


def _realistic_bg(n_ch, n, sfreq=1000.0, seed=0, uv=15.0):
    """Smooth, band-limited (1/f-like) EEG background ~``uv`` µV.

    White noise has unrealistically high per-sample slope; real (highpassed,
    notched) EEG is smooth, so injection is exercised against a background that
    resembles the real pre-trigger clean.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sfreq
    bg = np.zeros((n_ch, n), dtype=np.float64)
    for c in range(n_ch):
        for f in (2.0, 6.0, 10.0, 18.0):
            bg[c] += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    bg += 0.05 * rng.standard_normal((n_ch, n))
    bg = bg / np.max(np.abs(bg)) * (uv * 1e-6)
    return bg.astype(np.float32)


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


def test_inject_real_ieds_maps_by_name():
    ch_names = list(CH_NAMES)                      # Fp1,Fp2,F3,F4,C3,C4,P3,P4
    clean = _realistic_bg(len(ch_names), 5000, sfreq=1000.0, seed=0)  # ~15 uV smooth EEG
    pool = _fake_ied_pool(names=("C3", "C4", "F3"), focal_uv=300.0)
    out, centers = inject_real_ieds(
        clean, clean_sfreq=1000.0, ch_names=ch_names, ied_pool=pool, pool_sfreq=500.0,
        rate_hz=20.0, seed=1,
    )
    assert out.shape == clean.shape
    assert len(centers) > 0
    # spikes land on the named channels only (C3=4, C4=5, F3=2), never elsewhere
    hit_channels = {c for c, _ in centers}
    assert hit_channels <= {ch_names.index(n) for n in ("C3", "C4", "F3")}
    assert np.max(np.abs(out)) > 200e-6             # the 300 uV IED is there


def test_inject_real_ieds_skips_unmatched_names():
    ch_names = list(CH_NAMES)
    clean = np.zeros((len(ch_names), 4000), dtype=np.float32)
    pool = _fake_ied_pool(names=("PG1", "A1", "ECG1"))   # none present in CH_NAMES
    out, centers = inject_real_ieds(clean, 1000.0, ch_names, pool, 500.0, rate_hz=50.0, seed=2)
    assert len(centers) == 0          # nothing mapped -> nothing injected
    assert np.allclose(out, 0.0)


def test_inject_real_ieds_preserves_real_amplitude_and_topography():
    """No renormalisation: the real µV and the real channel ratios survive."""
    ch_names = list(CH_NAMES)
    pool = _fake_ied_pool(names=("C3", "C4", "F3"), focal_uv=300.0)  # ratios 1.0 / 0.5 / 0.3
    c3, c4, f3 = (ch_names.index(n) for n in ("C3", "C4", "F3"))
    clean = np.zeros((len(ch_names), 4000), dtype=np.float32)        # isolate the injection
    # Take the first seed that injects exactly one event (3 channels x 1 spike), so
    # the amplitude/ratio checks are not confounded by two overlapping IEDs.
    out = None
    for sd in range(60):
        cand, centers = inject_real_ieds(clean, 1000.0, ch_names, pool, 1000.0, rate_hz=0.5, seed=sd)
        if len(centers) == 3:
            out = cand
            break
    assert out is not None, "no seed produced a single isolated IED"
    peak_c3 = float(np.max(np.abs(out[c3]))) * 1e6
    assert 250.0 < peak_c3 < 305.0                                    # ~300 uV, not rescaled
    # real topography preserved (taper/baseline shift the values only marginally)
    assert np.isclose(np.max(np.abs(out[c4])) / np.max(np.abs(out[c3])), 0.5, atol=0.05)
    assert np.isclose(np.max(np.abs(out[f3])) / np.max(np.abs(out[c3])), 0.3, atol=0.05)


def test_inject_real_ieds_labels_every_injected_event_even_when_masked():
    """No visibility gate: an IED buried under a big background is still labelled.

    Gating on visibility would teach the model that a BCG-masked spike may be
    deleted, which is the opposite of spike preservation (run_3 §6.6).
    """
    ch_names = list(CH_NAMES)
    pool = _fake_ied_pool(names=("C3", "C4", "F3"), focal_uv=20.0)   # small IED
    huge_bg = _realistic_bg(len(ch_names), 6000, sfreq=1000.0, seed=5, uv=400.0)  # masks it
    _, centers = inject_real_ieds(huge_bg, 1000.0, ch_names, pool, 1000.0, rate_hz=3.0, seed=6)
    labelled = {c for c, _ in centers}
    assert labelled == {ch_names.index(n) for n in ("C3", "C4", "F3")}


def test_inject_real_ieds_does_not_add_a_step():
    """Baseline removal + taper: injection must not leave an edge discontinuity."""
    ch_names = list(CH_NAMES)
    t = 200
    # A waveform sitting on a large DC pedestal — untreated it would inject a step.
    wave = np.full(t, 500e-6)
    wave[t // 2] = 900e-6
    pool = [{"waveforms": np.stack([wave, wave, wave]).astype(np.float32),
             "names": ["C3", "C4", "F3"], "marker": t // 2}]
    clean = np.zeros((len(ch_names), 3000), dtype=np.float32)
    out, _ = inject_real_ieds(clean, 1000.0, ch_names, pool, 1000.0, rate_hz=0.4, seed=9)
    c3 = ch_names.index("C3")
    jumps = np.abs(np.diff(out[c3]))
    # the only large sample-to-sample change is the spike itself, not a pedestal edge
    assert np.sort(jumps)[-3] < 100e-6


def test_builder_real_ied_on_pretrigger_clean():
    # the key combination: REAL clean (pre-trigger surrogate) + REAL IED spikes
    bundle = _toy_bundle(n_epochs=40)
    n_ch = bundle["artifact"].shape[0]
    pre = _realistic_bg(n_ch, 800, sfreq=500.0, seed=3)
    ds = build_spatiotemporal_reference_dataset(
        bundle, context_epochs=7, core_samples=64, guard_samples=8,
        clean_source="niazy_pretrigger", pretrigger_clean=pre, pretrigger_sfreq=500.0,
        inject_spikes_mode=True, spike_source="real_ied",
        real_ied_pool=_fake_ied_pool(names=("C3", "C4", "F3")), real_ied_sfreq=500.0,
        spike_rate_hz=12.0, seed=0,
    )
    assert bool(ds["spikes_injected"][0]) is True
    assert str(ds["clean_source"][0]) == "niazy_pretrigger"
    assert float(np.sum(ds["spike_labels"])) > 0.0   # real IEDs produced ground-truth labels


def test_failure_modes_enrich_the_artifact_beyond_the_template():
    """run_3 §2: the artifact must NOT stay the epoch-invariant AAS template.

    Without this the target is exactly what AAS produces, so a perfect model can
    only reproduce AAS and has no in-band headroom.
    """
    from facet.training.spatiotemporal_builder import apply_artifact_failure_modes

    bundle = _toy_bundle(n_epochs=40, epoch_len=64)
    artifact = bundle["artifact"]
    sfreq = float(bundle["sfreq"][0])
    starts = bundle["triggers"][:-1]
    stops = bundle["triggers"][1:]

    enriched = apply_artifact_failure_modes(
        artifact, sfreq, starts, stops,
        epoch_amplitude_jitter=0.05, epoch_timing_jitter_samples=0.5,
        motion_drift_depth=0.05, motion_drift_hz=0.5,
        helium_pump_uv=2.0, helium_pump_hz=46.0, seed=0,
    )
    assert enriched.shape == artifact.shape
    assert not np.allclose(enriched, artifact)          # something actually changed

    # The template is epoch-periodic; the enriched artifact must NOT be, otherwise
    # epoch-averaging (i.e. AAS) could still remove it completely.
    ep = 64
    def _epoch_var(sig):
        blocks = sig[0, : (sig.shape[1] // ep) * ep].reshape(-1, ep)
        return float(np.mean(np.var(blocks, axis=0)))
    assert _epoch_var(enriched) > 10.0 * _epoch_var(artifact)

    # Disabling every mode is a no-op.
    same = apply_artifact_failure_modes(
        artifact, sfreq, starts, stops,
        epoch_amplitude_jitter=0.0, epoch_timing_jitter_samples=0.0,
        motion_drift_depth=0.0, helium_pump_uv=0.0, seed=0,
    )
    np.testing.assert_allclose(same, artifact, rtol=1e-5, atol=1e-8)


def test_builder_emits_leakage_free_split():
    """run_3 §7: train and val must not share centre epochs, and the guard band
    must separate the overlapping 7-epoch contexts."""
    bundle = _toy_bundle(n_epochs=60, epoch_len=40)
    ds = build_spatiotemporal_reference_dataset(
        bundle, context_epochs=7, core_samples=32, guard_samples=8,
        clean_source="synthetic", val_fraction=0.2, seed=0,
    )
    split = ds["example_split"]
    epochs = ds["center_epoch_index"]
    assert set(np.unique(split).tolist()) == {0, 1}
    train_ep = set(epochs[split == 0].tolist())
    val_ep = set(epochs[split == 1].tolist())
    assert not (train_ep & val_ep)                       # no shared centre epoch
    # guard: the 7-epoch contexts of the two sides must not overlap either
    assert min(val_ep) - max(train_ep) > 7 // 2


def test_builder_partitions_pretrigger_clean_disjointly():
    """The tiled clean must not repeat across the train/val boundary."""
    bundle = _toy_bundle(n_epochs=60, epoch_len=40)
    n_ch = bundle["artifact"].shape[0]
    # Distinctive pre-trigger clean: first half positive, second half negative, so
    # a leak across the boundary is directly visible in the sign.
    pre = np.ones((n_ch, 400), dtype=np.float32) * 1e-6
    pre[:, 200:] *= -1.0
    ds = build_spatiotemporal_reference_dataset(
        bundle, context_epochs=7, core_samples=32, guard_samples=8,
        clean_source="niazy_pretrigger", pretrigger_clean=pre, pretrigger_sfreq=500.0,
        val_fraction=0.5, failure_modes=False, seed=0,
    )
    split = ds["example_split"]
    clean_center = ds["clean_center"][:, 0, :]
    train_mean = float(np.mean(clean_center[split == 0]))
    val_mean = float(np.mean(clean_center[split == 1]))
    # train saw only the positive part of the source clean, val only the negative
    assert train_mean > 0 > val_mean


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
    # Enough epochs that the contiguous train/val split has a non-empty side even
    # after the 7-epoch guard band is dropped.
    bundle = _toy_bundle(n_epochs=40)
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
    # <= not ==: the leakage-free split deliberately drops the guard band at the seam
    assert len(train) > 0 and len(val) > 0
    assert len(train) + len(val) <= len(ds)
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
