"""Build a 7-epoch synthetic training dataset with variable artifact lengths.

Design goals:

1. fixed context width for the first model: exactly 7 epochs,
2. variable artifact duration and trigger delta in the source data,
3. fixed tensor shape for training by resampling each artifact epoch,
4. storage of original epoch lengths and phase features for later conditioning.

The output dataset is single-channel by default so that clean spike EEG and the
artifact source do not need to share the same montage.

Example:
    uv run python examples/build_synthetic_spike_artifact_context_dataset.py \
        --clean-file /path/to/clean_spike_recording.fif \
        --annotation-regex "spike|Spike|SPIKE" \
        --output-dir ./output/synthetic_spike_artifact_context_dataset
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from facet import load

DEFAULT_ARTIFACT_BUNDLE = Path("./output/niazy_artifact_extraction/niazy_estimated_artifact.npz")
DEFAULT_OUTPUT_DIR = Path("./output/synthetic_spike_artifact_context_dataset")
DEFAULT_ANNOTATION_REGEX = r"spike|Spike|SPIKE"
DEFAULT_CONTEXT_EPOCHS = 7
DEFAULT_ARTIFACT_SCALE_RANGE = (0.75, 1.25)
DEFAULT_MAX_JITTER_SAMPLES = 8
DEFAULT_SEED = 11
DEFAULT_ONSET_MIN_RATIO = 0.15
DEFAULT_ONSET_MAX_RATIO = 0.75


@dataclass
class ArtifactLibrarySource:
    source_id: str
    source_path: Path
    sfreq: float
    contexts: np.ndarray
    epoch_lengths: np.ndarray
    phase_linear: np.ndarray
    phase_sincos: np.ndarray

    @property
    def n_contexts(self) -> int:
        return int(self.contexts.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.contexts.shape[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-file", type=Path, required=True, help="Path to the clean annotated spike EEG recording")
    parser.add_argument(
        "--artifact-bundle",
        type=Path,
        default=DEFAULT_ARTIFACT_BUNDLE,
        help="Backward-compatible single artifact bundle path.",
    )
    parser.add_argument(
        "--artifact-library",
        type=Path,
        action="append",
        default=[],
        help=(
            "Artifact library source. Can be a .npz artifact bundle or a directory containing .npz bundles. "
            "May be provided multiple times. When omitted, --artifact-bundle is used."
        ),
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
        help="Directory for the synthetic context dataset",
    )
    parser.add_argument(
        "--context-epochs",
        type=int,
        default=DEFAULT_CONTEXT_EPOCHS,
        help="Number of consecutive epochs in the input context. Must be odd.",
    )
    parser.add_argument(
        "--target-epoch-samples",
        type=int,
        default=0,
        help="Resampled epoch length used by the model. Default: median artifact epoch length.",
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
        help="Maximum non-circular shift applied to each artifact epoch after resampling",
    )
    parser.add_argument(
        "--onset-min-ratio",
        type=float,
        default=DEFAULT_ONSET_MIN_RATIO,
        help="Minimum relative onset position within the center epoch for clean spike extraction.",
    )
    parser.add_argument(
        "--onset-max-ratio",
        type=float,
        default=DEFAULT_ONSET_MAX_RATIO,
        help="Maximum relative onset position within the center epoch for clean spike extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible sampling",
    )
    return parser.parse_args()


def _resample_1d(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.shape[-1] == target_length:
        return signal.astype(np.float32, copy=True)
    source_x = np.linspace(0.0, 1.0, num=signal.shape[-1], endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_x, source_x, signal).astype(np.float32, copy=False)


def _resample_multichannel(epoch: np.ndarray, target_length: int) -> np.ndarray:
    return np.vstack([_resample_1d(channel, target_length) for channel in epoch]).astype(np.float32, copy=False)


def _shift_with_edge_fill(epoch: np.ndarray, shift: int) -> np.ndarray:
    """Shift an epoch without circular wrap-around at the window boundaries."""
    if shift == 0:
        return epoch
    out = np.empty_like(epoch)
    if shift > 0:
        out[..., shift:] = epoch[..., :-shift]
        out[..., :shift] = epoch[..., :1]
    else:
        shift_abs = abs(shift)
        out[..., :-shift_abs] = epoch[..., shift_abs:]
        out[..., -shift_abs:] = epoch[..., -1:]
    return out


def _load_artifact_bundle(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact bundle not found at {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _discover_artifact_bundle_paths(paths: list[Path], fallback_bundle: Path) -> list[Path]:
    candidates = paths or [fallback_bundle]
    bundle_paths: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            bundle_paths.extend(sorted(path for path in candidate.rglob("*.npz") if path.is_file()))
        elif candidate.is_file():
            bundle_paths.append(candidate)
        else:
            raise FileNotFoundError(f"Artifact library source not found: {candidate}")

    unique_paths = list(dict.fromkeys(path.resolve() for path in bundle_paths))
    if not unique_paths:
        raise FileNotFoundError("No artifact .npz files found in artifact library sources")
    return unique_paths


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


def _artifact_epoch_boundaries(bundle: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    triggers = bundle["triggers"].astype(np.int64, copy=False)
    if triggers.size < 2:
        raise ValueError("At least two triggers are required to derive variable artifact epochs")

    offset_seconds = float(bundle["artifact_to_trigger_offset"][0])
    sfreq = float(bundle["sfreq"][0])
    offset_samples = int(round(offset_seconds * sfreq))

    starts = triggers[:-1] + offset_samples
    stops = triggers[1:] + offset_samples

    starts = np.clip(starts, 0, bundle["artifact"].shape[1] - 1)
    stops = np.clip(stops, 1, bundle["artifact"].shape[1])

    valid = stops > starts
    starts = starts[valid]
    stops = stops[valid]
    if starts.size == 0:
        raise ValueError("No valid variable artifact epochs could be derived from the trigger sequence")
    return starts.astype(np.int64, copy=False), stops.astype(np.int64, copy=False), offset_samples


def _build_artifact_epoch_library(
    bundle: dict[str, np.ndarray],
    target_epoch_samples: int,
    max_jitter_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    artifact = bundle["artifact"].astype(np.float32, copy=False)
    starts, stops, _offset_samples = _artifact_epoch_boundaries(bundle)

    lengths = (stops - starts).astype(np.int64, copy=False)
    resampled_epochs: list[np.ndarray] = []
    phase_linear: list[np.ndarray] = []
    phase_sincos: list[np.ndarray] = []

    for start, stop in zip(starts, stops, strict=False):
        epoch = artifact[:, start:stop]
        if epoch.shape[1] < 2:
            continue
        resampled_epochs.append(_resample_multichannel(epoch, target_epoch_samples))
        phase = np.linspace(0.0, 1.0, num=target_epoch_samples, endpoint=False, dtype=np.float32)
        phase_linear.append(phase[np.newaxis, :])
        phase_sincos.append(np.vstack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]).astype(np.float32))

    if not resampled_epochs:
        raise ValueError("No artifact epochs remained after resampling")

    epochs = np.stack(resampled_epochs, axis=0).astype(np.float32, copy=False)
    phase_linear_arr = np.stack(phase_linear, axis=0).astype(np.float32, copy=False)
    phase_sincos_arr = np.stack(phase_sincos, axis=0).astype(np.float32, copy=False)
    return epochs, lengths[: epochs.shape[0]], phase_linear_arr, phase_sincos_arr


def _build_artifact_contexts(
    artifact_epochs: np.ndarray,
    epoch_lengths: np.ndarray,
    phase_linear: np.ndarray,
    phase_sincos: np.ndarray,
    context_epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if context_epochs % 2 == 0 or context_epochs < 3:
        raise ValueError("context_epochs must be an odd integer >= 3")
    radius = context_epochs // 2
    if artifact_epochs.shape[0] <= 2 * radius:
        raise ValueError("Not enough artifact epochs to build the requested context size")

    contexts: list[np.ndarray] = []
    context_lengths: list[np.ndarray] = []
    context_phase_linear: list[np.ndarray] = []
    context_phase_sincos: list[np.ndarray] = []

    for center_idx in range(radius, artifact_epochs.shape[0] - radius):
        start = center_idx - radius
        stop = center_idx + radius + 1
        contexts.append(artifact_epochs[start:stop])
        context_lengths.append(epoch_lengths[start:stop])
        context_phase_linear.append(phase_linear[start:stop])
        context_phase_sincos.append(phase_sincos[start:stop])

    return (
        np.stack(contexts, axis=0).astype(np.float32, copy=False),
        np.stack(context_lengths, axis=0).astype(np.int64, copy=False),
        np.stack(context_phase_linear, axis=0).astype(np.float32, copy=False),
        np.stack(context_phase_sincos, axis=0).astype(np.float32, copy=False),
    )


def _build_artifact_library_sources(
    bundle_paths: list[Path],
    target_epoch_samples: int,
    context_epochs: int,
    max_jitter_samples: int,
) -> list[ArtifactLibrarySource]:
    sources: list[ArtifactLibrarySource] = []
    for path in bundle_paths:
        bundle = _load_artifact_bundle(path)
        artifact_epochs, epoch_lengths, phase_linear, phase_sincos = _build_artifact_epoch_library(
            bundle,
            target_epoch_samples=target_epoch_samples,
            max_jitter_samples=max_jitter_samples,
        )
        contexts, context_lengths, context_phase_linear, context_phase_sincos = _build_artifact_contexts(
            artifact_epochs,
            epoch_lengths,
            phase_linear,
            phase_sincos,
            context_epochs=context_epochs,
        )
        sources.append(
            ArtifactLibrarySource(
                source_id=path.stem,
                source_path=path,
                sfreq=float(bundle["sfreq"][0]),
                contexts=contexts,
                epoch_lengths=context_lengths,
                phase_linear=context_phase_linear,
                phase_sincos=context_phase_sincos,
            )
        )

    return sources


def _concatenate_artifact_sources(
    sources: list[ArtifactLibrarySource],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flattened_contexts: list[np.ndarray] = []
    flattened_lengths: list[np.ndarray] = []
    flattened_phase_linear: list[np.ndarray] = []
    flattened_phase_sincos: list[np.ndarray] = []
    flattened_source_ids: list[np.ndarray] = []
    flattened_source_paths: list[np.ndarray] = []
    flattened_channel_indices: list[np.ndarray] = []

    for source in sources:
        n_contexts, context_epochs, n_channels, epoch_samples = source.contexts.shape
        flattened_contexts.append(
            source.contexts.transpose(0, 2, 1, 3).reshape(n_contexts * n_channels, context_epochs, 1, epoch_samples)
        )
        flattened_lengths.append(np.repeat(source.epoch_lengths, n_channels, axis=0))
        flattened_phase_linear.append(np.repeat(source.phase_linear, n_channels, axis=0))
        flattened_phase_sincos.append(np.repeat(source.phase_sincos, n_channels, axis=0))
        flattened_source_ids.append(np.full(n_contexts * n_channels, source.source_id, dtype=object))
        flattened_source_paths.append(np.full(n_contexts * n_channels, str(source.source_path), dtype=object))
        flattened_channel_indices.append(np.tile(np.arange(n_channels, dtype=np.int64), n_contexts))

    contexts = np.concatenate(flattened_contexts, axis=0)
    lengths = np.concatenate(flattened_lengths, axis=0)
    phase_linear = np.concatenate(flattened_phase_linear, axis=0)
    phase_sincos = np.concatenate(flattened_phase_sincos, axis=0)
    source_ids = np.concatenate(
        flattened_source_ids,
        axis=0,
    )
    source_paths = np.concatenate(
        flattened_source_paths,
        axis=0,
    )
    source_channel_indices = np.concatenate(flattened_channel_indices, axis=0)
    return contexts, lengths, phase_linear, phase_sincos, source_ids, source_paths, source_channel_indices


def _extract_clean_contexts(
    raw: mne.io.BaseRaw,
    annotation_regex: str,
    context_epochs: int,
    epoch_samples: int,
    onset_min_ratio: float,
    onset_max_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= onset_min_ratio < onset_max_ratio < 1.0):
        raise ValueError("onset ratios must satisfy 0 <= min < max < 1")

    events, event_id = mne.events_from_annotations(raw, regexp=annotation_regex, verbose=False)
    if events.size == 0:
        raise ValueError(f"No annotations matched regex {annotation_regex!r}")

    descriptions_by_code = {value: key for key, value in event_id.items()}
    clean = raw.get_data().astype(np.float32, copy=False)
    radius = context_epochs // 2
    context_samples = context_epochs * epoch_samples

    contexts: list[np.ndarray] = []
    onset_offsets: list[int] = []
    descriptions: list[str] = []

    min_offset = int(round(onset_min_ratio * epoch_samples))
    max_offset = int(round(onset_max_ratio * epoch_samples))

    for sample, _zero, event_code in events:
        onset_offset = int(rng.integers(min_offset, max_offset + 1))
        context_start = int(sample) - (radius * epoch_samples + onset_offset)
        context_stop = context_start + context_samples
        if context_start < 0 or context_stop > clean.shape[1]:
            continue
        context = clean[:, context_start:context_stop].copy()
        contexts.append(context.reshape(clean.shape[0], context_epochs, epoch_samples).transpose(1, 0, 2))
        onset_offsets.append(onset_offset)
        descriptions.append(descriptions_by_code.get(int(event_code), "unknown"))

    if not contexts:
        raise ValueError("No valid clean 7-epoch contexts could be extracted from the annotated spike recording")

    return (
        np.stack(contexts, axis=0).astype(np.float32, copy=False),
        np.asarray(onset_offsets, dtype=np.int64),
        np.asarray(descriptions, dtype=object),
    )


def _build_single_channel_context_dataset(
    clean_contexts: np.ndarray,
    clean_channel_names: list[str],
    onset_offsets: np.ndarray,
    spike_descriptions: np.ndarray,
    artifact_contexts: np.ndarray,
    artifact_lengths: np.ndarray,
    phase_linear: np.ndarray,
    phase_sincos: np.ndarray,
    artifact_source_ids_by_context: np.ndarray,
    artifact_source_paths_by_context: np.ndarray,
    artifact_source_channel_indices_by_context: np.ndarray,
    artifact_scale_range: tuple[float, float],
    max_jitter_samples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    lo, hi = artifact_scale_range
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError("Artifact scale range must be positive and ordered")

    n_clean_examples, context_epochs, n_clean_channels, epoch_samples = clean_contexts.shape
    n_artifact_examples, _ctx_epochs, n_artifact_channels, _epoch_samples = artifact_contexts.shape
    if n_artifact_channels != 1:
        raise ValueError("Artifact contexts must be flattened to single-channel contexts before dataset construction")

    flat_clean = clean_contexts.transpose(0, 2, 1, 3).reshape(n_clean_examples * n_clean_channels, context_epochs, 1, epoch_samples)
    repeated_onset_offsets = np.repeat(onset_offsets, n_clean_channels)
    repeated_descriptions = np.repeat(spike_descriptions, n_clean_channels)
    repeated_channel_names = np.tile(np.asarray(clean_channel_names, dtype=object), n_clean_examples)

    artifact_indices = rng.integers(0, n_artifact_examples, size=flat_clean.shape[0])
    scales = rng.uniform(lo, hi, size=flat_clean.shape[0]).astype(np.float32)
    jitters = rng.integers(-max_jitter_samples, max_jitter_samples + 1, size=flat_clean.shape[0], dtype=np.int64)

    flat_artifact = np.empty_like(flat_clean)
    flat_phase_linear = np.empty((flat_clean.shape[0], context_epochs, 1, epoch_samples), dtype=np.float32)
    flat_phase_sincos = np.empty((flat_clean.shape[0], context_epochs, 2, epoch_samples), dtype=np.float32)
    selected_epoch_lengths = np.empty((flat_clean.shape[0], context_epochs), dtype=np.int64)

    for idx in range(flat_clean.shape[0]):
        context_idx = int(artifact_indices[idx])
        artifact_context = artifact_contexts[context_idx].astype(np.float32, copy=True)
        if max_jitter_samples > 0 and jitters[idx] != 0:
            artifact_context = _shift_with_edge_fill(artifact_context, int(jitters[idx]))
        flat_artifact[idx] = artifact_context * scales[idx]
        flat_phase_linear[idx] = phase_linear[context_idx]
        flat_phase_sincos[idx] = phase_sincos[context_idx]
        selected_epoch_lengths[idx] = artifact_lengths[context_idx]

    flat_noisy = flat_clean + flat_artifact
    center_idx = context_epochs // 2

    dataset = {
        "noisy_context": flat_noisy,
        "clean_context": flat_clean,
        "artifact_context": flat_artifact,
        "clean_center": flat_clean[:, center_idx],
        "artifact_center": flat_artifact[:, center_idx],
        "noisy_center": flat_noisy[:, center_idx],
        "trigger_phase_linear": flat_phase_linear,
        "trigger_phase_sincos": flat_phase_sincos,
        "artifact_epoch_lengths_samples": selected_epoch_lengths,
        "spike_onset_offsets_center_epoch": repeated_onset_offsets,
        "spike_descriptions": repeated_descriptions,
        "clean_channel_names": repeated_channel_names,
        "artifact_context_indices": artifact_indices.astype(np.int64, copy=False),
        "artifact_channel_indices": artifact_source_channel_indices_by_context[artifact_indices].astype(
            np.int64,
            copy=False,
        ),
        "artifact_source_ids": artifact_source_ids_by_context[artifact_indices].astype(object, copy=False),
        "artifact_source_paths": artifact_source_paths_by_context[artifact_indices].astype(object, copy=False),
        "artifact_scales": scales,
        "artifact_jitter_samples": jitters,
    }
    stats = {
        "n_examples": int(flat_noisy.shape[0]),
        "context_epochs": int(context_epochs),
        "epoch_samples": int(epoch_samples),
        "mean_abs_clean": float(np.mean(np.abs(flat_clean))),
        "mean_abs_artifact": float(np.mean(np.abs(flat_artifact))),
        "mean_abs_noisy": float(np.mean(np.abs(flat_noisy))),
        "min_artifact_epoch_length_samples": int(np.min(selected_epoch_lengths)),
        "max_artifact_epoch_length_samples": int(np.max(selected_epoch_lengths)),
        "median_artifact_epoch_length_samples": float(np.median(selected_epoch_lengths)),
        "n_artifact_sources": int(len(set(artifact_source_ids_by_context.tolist()))),
    }
    return dataset, stats


def _write_dataset(
    output_dir: Path,
    clean_file: Path,
    artifact_sources: list[ArtifactLibrarySource],
    dataset: dict[str, np.ndarray],
    sfreq: float,
    annotation_regex: str,
    stats: dict[str, int | float],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "synthetic_spike_artifact_context_dataset.npz"
    metadata_path = output_dir / "synthetic_spike_artifact_context_dataset_metadata.json"

    np.savez_compressed(dataset_path, sfreq=np.asarray([sfreq], dtype=np.float64), **dataset)

    metadata = {
        "clean_file": str(clean_file),
        "artifact_sources": [
            {
                "source_id": source.source_id,
                "path": str(source.source_path),
                "n_contexts": source.n_contexts,
                "n_channels": source.n_channels,
                "sampling_frequency_hz": source.sfreq,
            }
            for source in artifact_sources
        ],
        "dataset_path": str(dataset_path),
        "sampling_frequency_hz": float(sfreq),
        "annotation_regex": annotation_regex,
        "mode": "single_channel_context",
        "center_target": "artifact",
        **stats,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dataset_path, metadata_path


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.context_epochs % 2 == 0:
        raise ValueError("context-epochs must be odd so that a center epoch exists")

    artifact_bundle_paths = _discover_artifact_bundle_paths(args.artifact_library, args.artifact_bundle)
    first_bundle = _load_artifact_bundle(artifact_bundle_paths[0])
    target_sfreq = float(first_bundle["sfreq"][0])
    starts, stops, _offset = _artifact_epoch_boundaries(first_bundle)
    raw_lengths = (stops - starts).astype(np.int64, copy=False)
    target_epoch_samples = int(args.target_epoch_samples) if args.target_epoch_samples > 0 else int(np.median(raw_lengths))
    if target_epoch_samples < 8:
        raise ValueError("target epoch length must be at least 8 samples")

    artifact_sources = _build_artifact_library_sources(
        artifact_bundle_paths,
        target_epoch_samples=target_epoch_samples,
        context_epochs=args.context_epochs,
        max_jitter_samples=args.max_jitter_samples,
    )
    (
        artifact_contexts,
        artifact_context_lengths,
        artifact_phase_linear,
        artifact_phase_sincos,
        artifact_source_ids_by_context,
        artifact_source_paths_by_context,
        artifact_source_channel_indices_by_context,
    ) = _concatenate_artifact_sources(
        artifact_sources
    )

    clean_raw = _load_clean_raw(args.clean_file)
    clean_raw = _resample_if_needed(clean_raw, target_sfreq)
    clean_contexts, onset_offsets, spike_descriptions = _extract_clean_contexts(
        clean_raw,
        annotation_regex=args.annotation_regex,
        context_epochs=args.context_epochs,
        epoch_samples=target_epoch_samples,
        onset_min_ratio=args.onset_min_ratio,
        onset_max_ratio=args.onset_max_ratio,
        rng=rng,
    )

    dataset, stats = _build_single_channel_context_dataset(
        clean_contexts=clean_contexts,
        clean_channel_names=clean_raw.ch_names,
        onset_offsets=onset_offsets,
        spike_descriptions=spike_descriptions,
        artifact_contexts=artifact_contexts,
        artifact_lengths=artifact_context_lengths,
        phase_linear=artifact_phase_linear,
        phase_sincos=artifact_phase_sincos,
        artifact_source_ids_by_context=artifact_source_ids_by_context,
        artifact_source_paths_by_context=artifact_source_paths_by_context,
        artifact_source_channel_indices_by_context=artifact_source_channel_indices_by_context,
        artifact_scale_range=(args.artifact_scale_min, args.artifact_scale_max),
        max_jitter_samples=args.max_jitter_samples,
        rng=rng,
    )
    dataset_path, metadata_path = _write_dataset(
        output_dir=args.output_dir,
        clean_file=args.clean_file,
        artifact_sources=artifact_sources,
        dataset=dataset,
        sfreq=target_sfreq,
        annotation_regex=args.annotation_regex,
        stats=stats,
    )

    print("Saved synthetic 7-epoch spike-artifact context dataset:")
    print(f"  dataset  : {dataset_path}")
    print(f"  metadata : {metadata_path}")
    print(f"  examples : {stats['n_examples']}")
    print(f"  sources  : {stats['n_artifact_sources']}")
    print(f"  noisy    : {dataset['noisy_context'].shape}")
    print(f"  center   : {dataset['artifact_center'].shape}")
    print(
        f"  epoch lengths (min/median/max): {stats['min_artifact_epoch_length_samples']} / "
        f"{stats['median_artifact_epoch_length_samples']:.1f} / {stats['max_artifact_epoch_length_samples']}"
    )


if __name__ == "__main__":
    main()
