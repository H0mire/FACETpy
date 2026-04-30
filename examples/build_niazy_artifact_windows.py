"""Build a reusable artifact-window library from the extracted Niazy artifact bundle.

This is the next step after ``examples/extract_niazy_artifact_signal.py``:

1. load the saved artifact ``.npz`` bundle,
2. extract trigger-centered artifact windows,
3. write a compact window library that can later be mixed with clean spike EEG.

The output keeps the artifact windows in NumPy form because this is the most
practical intermediate format for synthetic dataset construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INPUT_BUNDLE = Path("./output/niazy_artifact_extraction/niazy_estimated_artifact.npz")
OUTPUT_DIR = Path("./output/niazy_artifact_windows")
WINDOWS_NPZ = OUTPUT_DIR / "niazy_artifact_windows.npz"
METADATA_JSON = OUTPUT_DIR / "niazy_artifact_windows_metadata.json"


def _load_bundle(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact bundle not found at {path}. Run examples/extract_niazy_artifact_signal.py first."
        )
    with np.load(path, allow_pickle=True) as bundle:
        return {key: bundle[key] for key in bundle.files}


def _derive_window_offsets(bundle: dict[str, np.ndarray]) -> tuple[int, int, int]:
    artifact_length = int(bundle["artifact_length"][0])
    if artifact_length <= 0:
        raise ValueError("artifact_length is missing or invalid in the artifact bundle")

    pre_trigger_samples = int(bundle.get("pre_trigger_samples", np.asarray([-1]))[0])
    post_trigger_samples = int(bundle.get("post_trigger_samples", np.asarray([-1]))[0])
    if pre_trigger_samples >= 0 and post_trigger_samples >= 0:
        return artifact_length, pre_trigger_samples, post_trigger_samples

    sfreq = float(bundle["sfreq"][0])
    offset_seconds = float(bundle["artifact_to_trigger_offset"][0])
    offset_samples = int(round(offset_seconds * sfreq))
    pre_trigger_samples = max(0, -offset_samples)
    post_trigger_samples = max(artifact_length - pre_trigger_samples, 0)
    if pre_trigger_samples + post_trigger_samples < artifact_length:
        post_trigger_samples = artifact_length - pre_trigger_samples
    return artifact_length, pre_trigger_samples, post_trigger_samples


def _extract_windows(bundle: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    artifact = bundle["artifact"].astype(np.float32, copy=False)
    triggers = bundle["triggers"].astype(np.int64, copy=False)
    if triggers.size == 0:
        raise ValueError("No triggers found in artifact bundle; cannot build trigger-aligned windows")

    artifact_length, pre_trigger_samples, post_trigger_samples = _derive_window_offsets(bundle)

    windows: list[np.ndarray] = []
    window_starts: list[int] = []
    n_samples = artifact.shape[1]

    for trigger in triggers:
        start = int(trigger) - pre_trigger_samples
        stop = start + artifact_length
        if start < 0 or stop > n_samples:
            continue
        windows.append(artifact[:, start:stop].copy())
        window_starts.append(start)

    if not windows:
        raise ValueError("No valid windows could be extracted from the artifact bundle")

    window_array = np.stack(windows, axis=0).astype(np.float32, copy=False)
    start_array = np.asarray(window_starts, dtype=np.int64)

    stats = {
        "artifact_length_samples": artifact_length,
        "pre_trigger_samples": pre_trigger_samples,
        "post_trigger_samples": post_trigger_samples,
        "n_windows": int(window_array.shape[0]),
        "n_channels": int(window_array.shape[1]),
        "window_samples": int(window_array.shape[2]),
        "mean_abs_amplitude": float(np.mean(np.abs(window_array))),
        "max_abs_amplitude": float(np.max(np.abs(window_array))),
    }
    return window_array, start_array, stats


def _write_outputs(
    bundle: dict[str, np.ndarray],
    windows: np.ndarray,
    starts: np.ndarray,
    stats: dict[str, int | float],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        WINDOWS_NPZ,
        artifact_windows=windows,
        window_start_samples=starts,
        trigger_samples=bundle["triggers"].astype(np.int64, copy=False),
        ch_names=bundle["ch_names"],
        sfreq=bundle["sfreq"].astype(np.float64, copy=False),
        artifact_length=bundle["artifact_length"].astype(np.int64, copy=False),
        artifact_to_trigger_offset=bundle["artifact_to_trigger_offset"].astype(np.float64, copy=False),
    )

    metadata = {
        "input_bundle": str(INPUT_BUNDLE),
        "output_npz": str(WINDOWS_NPZ),
        **stats,
        "channel_names": [str(ch) for ch in bundle["ch_names"].tolist()],
        "sampling_frequency_hz": float(bundle["sfreq"][0]),
        "trigger_count": int(bundle["triggers"].shape[0]),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    bundle = _load_bundle(INPUT_BUNDLE)
    windows, starts, stats = _extract_windows(bundle)
    _write_outputs(bundle, windows, starts, stats)

    print("Saved artifact window library:")
    print(f"  windows npz : {WINDOWS_NPZ}")
    print(f"  metadata    : {METADATA_JSON}")
    print(f"  windows     : {stats['n_windows']}")
    print(f"  shape       : {windows.shape}")
    print(f"  mean |amp|  : {stats['mean_abs_amplitude']:.6f}")
    print(f"  max |amp|   : {stats['max_abs_amplitude']:.6f}")


if __name__ == "__main__":
    main()
