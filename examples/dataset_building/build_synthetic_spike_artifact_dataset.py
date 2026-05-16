"""Mix clean spike EEG windows with Niazy artifact windows into a first training bundle.

Default design choice: single-channel training examples.
This avoids hard assumptions about matching montages between the clean spike EEG
recording and the Niazy fMRI artifact recording.

Example:
    uv run python examples/dataset_building/build_synthetic_spike_artifact_dataset.py \
        --clean-file /path/to/clean_spike_recording.fif \
        --annotation-regex "spike|Spike|SPIKE" \
        --output-dir ./output/synthetic_spike_artifact_dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

from facet import load

DEFAULT_ARTIFACT_WINDOWS = Path("./output/niazy_artifact_windows/niazy_artifact_windows.npz")
DEFAULT_OUTPUT_DIR = Path("./output/synthetic_spike_artifact_dataset")
DEFAULT_ANNOTATION_REGEX = r"spike|Spike|SPIKE"
DEFAULT_ARTIFACT_SCALE_RANGE = (0.75, 1.25)
DEFAULT_MAX_JITTER_SAMPLES = 8
DEFAULT_SEED = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-file", type=Path, required=True, help="Path to the clean annotated spike EEG recording")
    parser.add_argument(
        "--artifact-windows",
        type=Path,
        default=DEFAULT_ARTIFACT_WINDOWS,
        help="Path to niazy_artifact_windows.npz",
    )
    parser.add_argument(
        "--annotation-regex",
        type=str,
        default=DEFAULT_ANNOTATION_REGEX,
        help="Regular expression used to select spike annotations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the synthetic training bundle",
    )
    parser.add_argument(
        "--artifact-scale-min",
        type=float,
        default=DEFAULT_ARTIFACT_SCALE_RANGE[0],
        help="Lower bound for random artifact amplitude scaling",
    )
    parser.add_argument(
        "--artifact-scale-max",
        type=float,
        default=DEFAULT_ARTIFACT_SCALE_RANGE[1],
        help="Upper bound for random artifact amplitude scaling",
    )
    parser.add_argument(
        "--max-jitter-samples",
        type=int,
        default=DEFAULT_MAX_JITTER_SAMPLES,
        help="Maximum circular shift applied to each sampled artifact window",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible artifact sampling",
    )
    return parser.parse_args()


def _load_artifact_windows(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact window library not found at {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_clean_raw(path: Path) -> mne.io.BaseRaw:
    context = load(str(path), preload=True)
    raw = context.get_raw().copy()
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    if len(eeg_picks) == 0:
        raise ValueError("The clean recording does not contain EEG channels")
    return raw.pick(eeg_picks)


def _resample_if_needed(raw: mne.io.BaseRaw, target_sfreq: float) -> mne.io.BaseRaw:
    current_sfreq = float(raw.info["sfreq"])
    if np.isclose(current_sfreq, target_sfreq):
        return raw
    out = raw.copy()
    out.resample(target_sfreq, npad="auto", verbose=False)
    return out


def _extract_spike_windows(
    raw: mne.io.BaseRaw,
    annotation_regex: str,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    events, event_id = mne.events_from_annotations(raw, regexp=annotation_regex, verbose=False)
    if events.size == 0:
        raise ValueError(f"No annotations matched regex {annotation_regex!r} in {raw.filenames[0] if raw.filenames else 'raw'}")

    descriptions_by_code = {value: key for key, value in event_id.items()}
    half_window = window_size // 2
    clean = raw.get_data().astype(np.float32, copy=False)

    windows: list[np.ndarray] = []
    centers: list[int] = []
    descriptions: list[str] = []

    for sample, _zero, event_code in events:
        start = int(sample) - half_window
        stop = start + window_size
        if start < 0 or stop > clean.shape[1]:
            continue
        windows.append(clean[:, start:stop].copy())
        centers.append(int(sample))
        descriptions.append(descriptions_by_code.get(int(event_code), "unknown"))

    if not windows:
        raise ValueError("No valid spike-centered windows fit into the clean recording")

    return (
        np.stack(windows, axis=0).astype(np.float32, copy=False),
        np.asarray(centers, dtype=np.int64),
        np.asarray(descriptions, dtype=object),
    )


def _build_single_channel_dataset(
    clean_windows: np.ndarray,
    clean_channel_names: list[str],
    spike_centers: np.ndarray,
    spike_descriptions: np.ndarray,
    artifact_windows: np.ndarray,
    rng: np.random.Generator,
    artifact_scale_range: tuple[float, float],
    max_jitter_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    n_spikes, n_clean_channels, window_size = clean_windows.shape
    lo, hi = artifact_scale_range
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError("Artifact scale range must be positive and ordered")

    flat_clean = clean_windows.reshape(n_spikes * n_clean_channels, 1, window_size).astype(np.float32, copy=False)
    repeated_centers = np.repeat(spike_centers, n_clean_channels)
    repeated_descriptions = np.repeat(spike_descriptions, n_clean_channels)
    repeated_channel_names = np.tile(np.asarray(clean_channel_names, dtype=object), n_spikes)

    n_artifact_windows, n_artifact_channels, _artifact_size = artifact_windows.shape
    artifact_indices = rng.integers(0, n_artifact_windows, size=flat_clean.shape[0])
    artifact_channel_indices = rng.integers(0, n_artifact_channels, size=flat_clean.shape[0])
    scales = rng.uniform(lo, hi, size=flat_clean.shape[0]).astype(np.float32)
    jitters = rng.integers(-max_jitter_samples, max_jitter_samples + 1, size=flat_clean.shape[0], dtype=np.int64)

    flat_artifact = np.empty_like(flat_clean)
    for idx in range(flat_clean.shape[0]):
        artifact = artifact_windows[artifact_indices[idx], artifact_channel_indices[idx]].astype(np.float32, copy=True)
        if max_jitter_samples > 0 and jitters[idx] != 0:
            artifact = np.roll(artifact, int(jitters[idx]))
        flat_artifact[idx, 0] = artifact * scales[idx]

    flat_noisy = flat_clean + flat_artifact

    dataset = {
        "clean": flat_clean,
        "artifact": flat_artifact,
        "noisy": flat_noisy,
        "spike_center_samples": repeated_centers,
        "spike_descriptions": repeated_descriptions,
        "clean_channel_names": repeated_channel_names,
        "artifact_window_indices": artifact_indices.astype(np.int64, copy=False),
        "artifact_channel_indices": artifact_channel_indices.astype(np.int64, copy=False),
        "artifact_scales": scales,
        "artifact_jitter_samples": jitters,
    }
    stats = {
        "n_examples": int(flat_clean.shape[0]),
        "n_spike_windows": int(n_spikes),
        "n_clean_channels": int(n_clean_channels),
        "window_size_samples": int(window_size),
        "mean_abs_clean": float(np.mean(np.abs(flat_clean))),
        "mean_abs_artifact": float(np.mean(np.abs(flat_artifact))),
        "mean_abs_noisy": float(np.mean(np.abs(flat_noisy))),
    }
    return dataset, stats


def _write_dataset(
    output_dir: Path,
    clean_file: Path,
    artifact_windows_path: Path,
    dataset: dict[str, np.ndarray],
    sfreq: float,
    annotation_regex: str,
    stats: dict[str, int | float],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "synthetic_spike_artifact_dataset.npz"
    metadata_path = output_dir / "synthetic_spike_artifact_dataset_metadata.json"

    np.savez_compressed(
        dataset_path,
        sfreq=np.asarray([sfreq], dtype=np.float64),
        **dataset,
    )

    metadata = {
        "clean_file": str(clean_file),
        "artifact_windows": str(artifact_windows_path),
        "dataset_path": str(dataset_path),
        "sampling_frequency_hz": float(sfreq),
        "annotation_regex": annotation_regex,
        "mode": "single_channel",
        **stats,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dataset_path, metadata_path


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    artifact_library = _load_artifact_windows(args.artifact_windows)
    artifact_windows = artifact_library["artifact_windows"].astype(np.float32, copy=False)
    target_sfreq = float(artifact_library["sfreq"][0])
    window_size = int(artifact_windows.shape[2])

    clean_raw = _load_clean_raw(args.clean_file)
    clean_raw = _resample_if_needed(clean_raw, target_sfreq)
    clean_windows, spike_centers, spike_descriptions = _extract_spike_windows(
        clean_raw,
        annotation_regex=args.annotation_regex,
        window_size=window_size,
    )

    dataset, stats = _build_single_channel_dataset(
        clean_windows=clean_windows,
        clean_channel_names=clean_raw.ch_names,
        spike_centers=spike_centers,
        spike_descriptions=spike_descriptions,
        artifact_windows=artifact_windows,
        rng=rng,
        artifact_scale_range=(args.artifact_scale_min, args.artifact_scale_max),
        max_jitter_samples=args.max_jitter_samples,
    )
    dataset_path, metadata_path = _write_dataset(
        output_dir=args.output_dir,
        clean_file=args.clean_file,
        artifact_windows_path=args.artifact_windows,
        dataset=dataset,
        sfreq=target_sfreq,
        annotation_regex=args.annotation_regex,
        stats=stats,
    )

    print("Saved synthetic spike-artifact training bundle:")
    print(f"  dataset  : {dataset_path}")
    print(f"  metadata : {metadata_path}")
    print(f"  examples : {stats['n_examples']}")
    print(f"  shape    : {dataset['noisy'].shape}")
    print(f"  sfreq    : {target_sfreq} Hz")


if __name__ == "__main__":
    main()
