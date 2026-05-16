"""Generate a synthetic fMRI gradient-artifact source bundle.

This creates an artifact-only source compatible with
``examples/dataset_building/build_synthetic_spike_artifact_context_dataset.py``. The generated
signal approximates the properties that matter for the training data:

* deterministic and trigger-locked morphology,
* repeated slice/readout substructure within each artifact epoch,
* channel-specific gain, polarity, delay and ringing,
* slow epoch-to-epoch drift rather than independent random shapes,
* explicit artifact onset/offset so epochs do not look like continuous noise.

It is still not a scanner-accurate Bloch/gradient-coil simulation. Its purpose
is a plausible domain-randomization source, not a replacement for measured
AAS-derived artifact templates.

Example:
    uv run python examples/dataset_building/generate_synthetic_fmri_artifact_source.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

OUTPUT_DIR = Path("./output/artifact_libraries/generated_fmri")
ARTIFACT_NPZ = OUTPUT_DIR / "generated_fmri_gradient_artifact.npz"
ARTIFACT_FIF = OUTPUT_DIR / "generated_fmri_gradient_artifact_raw.fif"
METADATA_JSON = OUTPUT_DIR / "generated_fmri_gradient_artifact_metadata.json"


def _smooth(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Small Gaussian smoother used to mimic finite amplifier bandwidth."""
    if sigma_samples <= 0:
        return values
    radius = max(1, int(round(4 * sigma_samples)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def _gaussian_lobe(phase: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((phase - center) / width) ** 2)


def _epi_slice_waveform(phase: np.ndarray, *, readout_cycles: int) -> np.ndarray:
    """Create one slice-scale EPI-like artifact component.

    The waveform intentionally contains sharp bipolar switching lobes and an
    alternating readout train. EEG gradient artifacts are induced by changing
    magnetic gradients, so this shape is closer to a switched-gradient sequence
    than random Gaussian peaks.
    """
    phase = np.clip(phase, 0.0, 1.0)
    wave = np.zeros_like(phase, dtype=np.float64)

    # Slice-select and phase-encode switching lobes.
    wave += _gaussian_lobe(phase, 0.08, 0.018, 1.05)
    wave += _gaussian_lobe(phase, 0.16, 0.018, -0.95)
    wave += _gaussian_lobe(phase, 0.27, 0.022, 0.50)

    # Alternating EPI readout train with a soft envelope. The tanh-shaped
    # waveform keeps the periodic switching visible without sample-wise noise.
    readout_start = 0.31
    readout_stop = 0.86
    readout_phase = (phase - readout_start) / (readout_stop - readout_start)
    readout_mask = (readout_phase >= 0.0) & (readout_phase < 1.0)
    local = np.clip(readout_phase, 0.0, 1.0)
    envelope = np.sin(np.pi * local) ** 0.35
    readout_carrier = np.tanh(3.2 * np.sin(2.0 * np.pi * readout_cycles * local))
    sinusoidal_edge = 0.18 * np.sin(2.0 * np.pi * (2 * readout_cycles) * local + 0.35)
    wave += readout_mask * envelope * (1.05 * readout_carrier + sinusoidal_edge)

    # Rephasing lobe and amplifier/ring-down tail.
    wave += _gaussian_lobe(phase, 0.89, 0.030, -0.85)
    tail = np.maximum(phase - 0.88, 0.0)
    wave += 0.20 * np.sin(2.0 * np.pi * 18.0 * tail) * np.exp(-18.0 * tail) * (phase >= 0.88)
    return wave


def _base_gradient_template(n_samples: int, *, n_slices: int = 21) -> np.ndarray:
    """Create one deterministic volume/slice artifact template."""
    sample_phase = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    template = np.zeros(n_samples, dtype=np.float64)

    active_start = 0.055
    active_stop = 0.925
    active_span = active_stop - active_start

    for slice_idx in range(n_slices):
        start = active_start + active_span * slice_idx / n_slices
        stop = active_start + active_span * (slice_idx + 1) / n_slices
        local_phase = (sample_phase - start) / (stop - start)
        mask = (local_phase >= 0.0) & (local_phase < 1.0)
        if not np.any(mask):
            continue
        # Keep all slices highly similar. Device variation should happen across
        # generated sources/channels, not as random-looking within-epoch noise.
        readout_cycles = 5
        slice_gain = 0.96 + 0.04 * np.sin(2.0 * np.pi * slice_idx / n_slices)
        template[mask] += slice_gain * _epi_slice_waveform(
            local_phase[mask],
            readout_cycles=readout_cycles,
        )

    # Deterministic scanner vibration component, gated to the acquisition block.
    active = (sample_phase >= active_start) & (sample_phase <= active_stop)
    template += active * 0.06 * np.sin(2.0 * np.pi * 92.0 * sample_phase + 0.4)

    # Clear begin/end: outside the gradient train the artifact should be quiet.
    fade = np.zeros_like(sample_phase, dtype=np.float64)
    fade_in_stop = active_start + 0.025
    fade_out_start = active_stop - 0.025
    fade[(sample_phase >= fade_in_stop) & (sample_phase <= fade_out_start)] = 1.0
    in_mask = (sample_phase >= active_start) & (sample_phase < fade_in_stop)
    out_mask = (sample_phase > fade_out_start) & (sample_phase <= active_stop)
    fade[in_mask] = 0.5 - 0.5 * np.cos(np.pi * (sample_phase[in_mask] - active_start) / (fade_in_stop - active_start))
    fade[out_mask] = 0.5 + 0.5 * np.cos(np.pi * (sample_phase[out_mask] - fade_out_start) / (active_stop - fade_out_start))

    template *= fade
    template = _smooth(template, sigma_samples=0.70)

    # Preserve a true quiet baseline. A global mean subtraction would lift the
    # non-artifact background to an artificial offset that can look like noise.
    active_mask = fade > 0.05
    if np.any(active_mask):
        active_mean = float(np.mean(template[active_mask]))
        template = (template - active_mean) * fade

    peak = float(np.max(np.abs(template)))
    if peak > 0:
        template /= peak
    return template.astype(np.float32)


def _resample_1d(values: np.ndarray, target_samples: int) -> np.ndarray:
    if len(values) == target_samples:
        return values.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, len(values), endpoint=False)
    x_new = np.linspace(0.0, 1.0, target_samples, endpoint=False)
    return np.interp(x_new, x_old, values).astype(np.float32)


def _shift_with_edge_fill(values: np.ndarray, shift: int) -> np.ndarray:
    """Shift without wrapping the opposite artifact edge into the epoch."""
    if shift == 0:
        return values
    out = np.empty_like(values)
    if shift > 0:
        out[shift:] = values[:-shift]
        out[:shift] = values[0]
    else:
        shift_abs = abs(shift)
        out[:-shift_abs] = values[shift_abs:]
        out[-shift_abs:] = values[-1]
    return out


def generate_artifact_source(
    *,
    sfreq: float = 2048.0,
    n_channels: int = 30,
    n_epochs: int = 900,
    median_epoch_samples: int = 292,
    max_epoch_jitter_samples: int = 11,
    amplitude_uv: float = 5000.0,
    seed: int = 20260430,
) -> tuple[mne.io.BaseRaw, dict[str, object]]:
    rng = np.random.default_rng(seed)
    deltas = median_epoch_samples + rng.integers(
        0,
        max_epoch_jitter_samples + 1,
        size=n_epochs,
        dtype=np.int64,
    )
    triggers = np.concatenate([[0], np.cumsum(deltas)]).astype(np.int64)
    n_samples = int(triggers[-1])
    artifact = np.zeros((n_channels, n_samples), dtype=np.float32)

    max_template_samples = median_epoch_samples + max_epoch_jitter_samples
    base_template = _base_gradient_template(max_template_samples)

    channel_gains = rng.lognormal(mean=0.0, sigma=0.28, size=n_channels).astype(np.float32)
    channel_polarities = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n_channels)
    channel_delays = rng.integers(-1, 2, size=n_channels)
    ring_freqs = rng.uniform(70.0, 100.0, size=n_channels)
    ring_phases = rng.uniform(0.0, 2.0 * np.pi, size=n_channels)
    ring_gains = rng.uniform(0.0, 0.004, size=n_channels)

    for epoch_idx in range(n_epochs):
        start = int(triggers[epoch_idx])
        stop = int(triggers[epoch_idx + 1])
        length = stop - start
        base = _resample_1d(base_template, length)
        drift_phase = 2.0 * np.pi * epoch_idx / max(n_epochs, 1)
        epoch_scale = float(1.0 + 0.04 * np.sin(2.7 * drift_phase) + rng.normal(0.0, 0.004))
        epoch_delay = 0
        if epoch_delay:
            base = _shift_with_edge_fill(base, epoch_delay)
        phase = np.arange(length, dtype=np.float32) / sfreq
        for ch_idx in range(n_channels):
            shifted = _shift_with_edge_fill(base, int(channel_delays[ch_idx]))
            ringing = ring_gains[ch_idx] * np.sin(2.0 * np.pi * ring_freqs[ch_idx] * phase + ring_phases[ch_idx])
            ringing *= np.exp(-phase / 0.035)
            # Background is deliberately ~10^3 to 10^4 below the artifact peak.
            channel_noise = rng.normal(0.0, 0.000002, size=length).astype(np.float32)
            artifact[ch_idx, start:stop] = (
                (amplitude_uv * 1e-6)
                * epoch_scale
                * channel_gains[ch_idx]
                * channel_polarities[ch_idx]
                * (shifted + channel_noise)
            )

    ch_names = [f"EEG{i + 1:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    raw = mne.io.RawArray(artifact, info, verbose=False)

    metadata = {
        "artifact_npz": str(ARTIFACT_NPZ),
        "artifact_fif": str(ARTIFACT_FIF),
        "source_type": "synthetic_fmri_gradient_artifact",
        "n_channels": n_channels,
        "n_samples": n_samples,
        "n_epochs": n_epochs,
        "sfreq": sfreq,
        "base_epoch_samples": median_epoch_samples,
        "median_epoch_samples": float(np.median(deltas)),
        "min_epoch_samples": int(deltas.min()),
        "max_epoch_samples": int(deltas.max()),
        "amplitude_uv": amplitude_uv,
        "n_slices": 21,
        "generator_model": "deterministic_epi_slice_readout_template_with_channel_delay_gain_and_ringing",
        "seed": seed,
    }
    return raw, {"metadata": metadata, "triggers": triggers}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, payload = generate_artifact_source()
    triggers = payload["triggers"]
    metadata = payload["metadata"]

    raw.save(ARTIFACT_FIF, overwrite=True, verbose=False)
    np.savez_compressed(
        ARTIFACT_NPZ,
        artifact=raw.get_data().astype(np.float32, copy=False),
        corrected=np.zeros(raw.get_data().shape, dtype=np.float32),
        ch_names=np.asarray(raw.ch_names, dtype=object),
        sfreq=np.asarray([raw.info["sfreq"]], dtype=np.float64),
        triggers=triggers.astype(np.int64, copy=False),
        artifact_length=np.asarray([-1], dtype=np.int64),
        artifact_to_trigger_offset=np.asarray([0.0], dtype=np.float64),
        acq_start_sample=np.asarray([0], dtype=np.int64),
        acq_end_sample=np.asarray([raw.n_times], dtype=np.int64),
        pre_trigger_samples=np.asarray([0], dtype=np.int64),
        post_trigger_samples=np.asarray([0], dtype=np.int64),
    )
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved synthetic fMRI artifact source:")
    print(f"  artifact npz : {ARTIFACT_NPZ}")
    print(f"  artifact raw : {ARTIFACT_FIF}")
    print(f"  metadata json: {METADATA_JSON}")
    print(f"  channels     : {metadata['n_channels']}")
    print(f"  epochs       : {metadata['n_epochs']}")
    print(
        "  epoch samples: "
        f"{metadata['min_epoch_samples']} / {metadata['median_epoch_samples']} / {metadata['max_epoch_samples']}"
    )


if __name__ == "__main__":
    main()
