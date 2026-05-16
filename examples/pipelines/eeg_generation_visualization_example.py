"""
Synthetic EEG generation and visualization.

Useful for testing your pipeline settings before running on real data.
EEGGenerator creates a realistic multi-channel EEG recording with neural
oscillations, physiological artifacts, and optional epileptic spikes.

Run this example independently of any real dataset — no EDF file needed.

Sections:
  A. Standard synthetic EEG (default oscillation + noise parameters)
  B. Adjust oscillation and noise parameters
  C. Add simulated epileptic spike activity
"""

from pathlib import Path

from facet import (
    Pipeline,
    HighPassFilter,
    LowPassFilter,
    RawPlotter,
    EEGGenerator,
)

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# A. Standard synthetic recording
# ---------------------------------------------------------------------------
# EEGGenerator is the first step — no EDF file or initial data needed.
# Default settings give 32 EEG + 2 EOG + 1 ECG channels at 1 kHz.

pipeline_a = Pipeline([
    EEGGenerator(
        sampling_rate=1000,
        duration=30.0,
        random_seed=42,          # fixed seed for reproducibility
    ),
    HighPassFilter(freq=1.0),
    RawPlotter(
        mode="matplotlib",
        channel="Fp1",
        start=5.0,
        duration=15.0,
        save_path=str(OUTPUT_DIR / "synthetic_eeg_standard.png"),
        show=False,
        auto_close=True,
        title="Synthetic EEG — Standard Parameters",
    ),
], name="Standard Synthetic EEG")

result_a = pipeline_a.run()
result_a.print_summary()

raw = result_a.context.get_raw()
print(f"  Channels: {raw.info['nchan']}, "
      f"Duration: {raw.n_times / raw.info['sfreq']:.1f}s, "
      f"Sampling rate: {raw.info['sfreq']:.0f} Hz")


# ---------------------------------------------------------------------------
# B. Custom oscillation and noise parameters
# ---------------------------------------------------------------------------
# Researchers studying specific frequency bands can amplify those components.

pipeline_b = Pipeline([
    EEGGenerator(
        sampling_rate=1000,
        duration=20.0,
        channel_schema={"eeg_channels": 32, "eog_channels": 2, "ecg_channels": 1},
        oscillation_params={
            "alpha": 25.0,   # stronger alpha (e.g. eyes-closed resting state)
            "beta":  8.0,
            "gamma": 3.0,
        },
        noise_params={
            "pink_noise_amplitude": 15.0,
            "white_noise_amplitude": 2.0,
            "line_noise_amplitude": 1.0,   # 50 Hz line noise
        },
        random_seed=42,
    ),
    HighPassFilter(freq=1.0),
    LowPassFilter(freq=70.0),
    RawPlotter(
        mode="matplotlib",
        channel="Fp1",
        start=0.0,
        duration=15.0,
        save_path=str(OUTPUT_DIR / "synthetic_eeg_custom.png"),
        show=False,
        auto_close=True,
        title="Synthetic EEG — Dominant Alpha",
    ),
], name="Custom Oscillation EEG")

result_b = pipeline_b.run()
result_b.print_summary()


# ---------------------------------------------------------------------------
# C. Epileptic spike activity
# ---------------------------------------------------------------------------
# Enable spikes to test spike-detection algorithms or verify that your
# correction pipeline does not introduce or remove spike-like transients.

pipeline_c = Pipeline([
    EEGGenerator(
        sampling_rate=1000,
        duration=60.0,
        channel_schema={"eeg_channels": 64, "eog_channels": 2, "ecg_channels": 1},
        spike_params={
            "enabled": True,
            "spike_rate": 15.0,          # spikes per minute
            "spike_amplitude": 150.0,    # μV
            "spike_duration_ms": 50.0,
        },
        random_seed=42,
    ),
    RawPlotter(
        mode="matplotlib",
        channel="Fp1",
        start=10.0,
        duration=20.0,
        save_path=str(OUTPUT_DIR / "synthetic_eeg_spikes.png"),
        show=False,
        auto_close=True,
        title="Synthetic EEG — With Epileptic Spikes",
    ),
], name="Synthetic EEG with Spikes")

result_c = pipeline_c.run()

# Read how many spikes were generated from the metadata
n_spikes = result_c.context.metadata.custom.get("eeg_generator", {}).get("n_spikes", 0)
print(f"\nSection C — generated {n_spikes} epileptic spike events")
result_c.print_summary()
