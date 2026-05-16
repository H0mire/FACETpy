"""Generate a clean synthetic spike EEG source dataset for training experiments.

The generated recording contains synthetic EEG with epileptiform spike activity.
Spike onsets are written as MNE annotations, so the file can be used directly
with ``examples/dataset_building/build_synthetic_spike_artifact_context_dataset.py``.

Example:
    uv run python examples/dataset_building/generate_synthetic_spike_source_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

from facet import EEGGenerator, Pipeline

OUTPUT_DIR = Path("./output/synthetic_spike_source")
OUTPUT_RAW = OUTPUT_DIR / "synthetic_spike_source_raw.fif"
OUTPUT_METADATA = OUTPUT_DIR / "synthetic_spike_source_metadata.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = Pipeline(
        [
            EEGGenerator(
                sampling_rate=2048,
                duration=300.0,
                channel_schema={
                    "eeg_channels": 31,
                    "eog_channels": 0,
                    "ecg_channels": 0,
                    "emg_channels": 0,
                    "misc_channels": 0,
                },
                artifact_params={
                    "blink_rate": 0.0,
                    "saccade_rate": 0.0,
                    "heart_rate": 70.0,
                    "emg_amplitude": 0.0,
                },
                noise_params={
                    "pink_noise_amplitude": 8.0,
                    "white_noise_amplitude": 0.8,
                    "line_noise_amplitude": 0.2,
                },
                spike_params={
                    "enabled": True,
                    "spike_rate": 12.0,
                    "spike_amplitude": 140.0,
                    "spike_duration_ms": 45.0,
                    "slow_wave_enabled": True,
                    "slow_wave_amplitude_ratio": 0.45,
                    "slow_wave_duration_ms": 180.0,
                    "polyspike_probability": 0.15,
                    "spatial_spread": 0.45,
                    "annotate": True,
                    "annotation_description": "spike_onset",
                },
                random_seed=123,
            )
        ],
        name="Synthetic spike source dataset",
    ).run(show_progress=False)

    raw = result.context.get_raw()
    raw.save(OUTPUT_RAW, overwrite=True, verbose=False)

    generator_meta = result.context.metadata.custom.get("eeg_generator", {})
    metadata = {
        "raw_file": str(OUTPUT_RAW),
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "n_channels": int(raw.info["nchan"]),
        "n_samples": int(raw.n_times),
        "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
        "annotation_description": generator_meta.get("spike_annotation_description", "spike_onset"),
        "n_spikes": int(generator_meta.get("n_spikes", 0)),
        "spike_events": generator_meta.get("spike_events", []),
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved synthetic spike source dataset:")
    print(f"  raw      : {OUTPUT_RAW}")
    print(f"  metadata : {OUTPUT_METADATA}")
    print(f"  spikes   : {metadata['n_spikes']}")
    print(f"  sfreq    : {metadata['sampling_frequency_hz']} Hz")
    print(f"  channels : {metadata['n_channels']}")


if __name__ == "__main__":
    main()
