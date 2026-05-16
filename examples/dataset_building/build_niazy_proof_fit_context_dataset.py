"""Build a Niazy-only proof-fit context dataset.

This dataset is intentionally not a clean generalization benchmark. It uses the
same Niazy recording as both the training source and the later inference source
to answer a narrower question first:

Can the model family learn the AAS-estimated artifact morphology at all?

The clean target is therefore a surrogate:

- ``clean_context``: AAS-corrected Niazy EEG from the artifact bundle
- ``artifact_context``: AAS-estimated artifact from the same bundle
- ``noisy_context``: ``clean_context + artifact_context``

The output NPZ follows the same contract as the synthetic context datasets used
by ``cascaded_dae`` and ``cascaded_context_dae``.

Example:
    uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py \
        --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
        --target-epoch-samples 512 \
        --context-epochs 7 \
        --output-dir output/niazy_proof_fit_context_512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_ARTIFACT_BUNDLE = Path("output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz")
DEFAULT_OUTPUT_DIR = Path("output/niazy_proof_fit_context_512")
DATASET_FILENAME = "niazy_proof_fit_context_dataset.npz"
METADATA_FILENAME = "niazy_proof_fit_context_dataset_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-bundle", type=Path, default=DEFAULT_ARTIFACT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-epochs", type=int, default=7)
    parser.add_argument("--target-epoch-samples", type=int, default=512)
    return parser.parse_args()


def _resample_1d(values: np.ndarray, target_samples: int) -> np.ndarray:
    if values.shape[-1] == target_samples:
        return values.astype(np.float32, copy=True)
    if values.shape[-1] == 0:
        return np.zeros(target_samples, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, values.shape[-1], endpoint=False, dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_samples, endpoint=False, dtype=np.float64)
    return np.interp(target_x, source_x, values).astype(np.float32)


def _resample_epoch(epoch: np.ndarray, target_samples: int) -> np.ndarray:
    return np.vstack([_resample_1d(channel, target_samples) for channel in epoch]).astype(np.float32)


def _epoch_boundaries(bundle: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    triggers = np.asarray(bundle["triggers"], dtype=np.int64)
    if triggers.size < 2:
        raise ValueError("At least two triggers are required")

    sfreq = float(bundle["sfreq"][0])
    offset_seconds = float(bundle["artifact_to_trigger_offset"][0])
    offset_samples = int(round(offset_seconds * sfreq))
    n_samples = int(bundle["artifact"].shape[1])

    starts = triggers[:-1] + offset_samples
    stops = triggers[1:] + offset_samples
    valid = (starts >= 0) & (stops > starts) & (stops <= n_samples)
    starts = starts[valid].astype(np.int64)
    stops = stops[valid].astype(np.int64)
    if starts.size == 0:
        raise ValueError("No valid trigger-to-trigger epochs remained after clipping")
    return starts, stops


def _build_context_dataset(
    bundle: dict[str, np.ndarray],
    *,
    context_epochs: int,
    target_epoch_samples: int,
) -> dict[str, np.ndarray]:
    if context_epochs < 3 or context_epochs % 2 == 0:
        raise ValueError("context_epochs must be an odd integer >= 3")
    if target_epoch_samples < 8:
        raise ValueError("target_epoch_samples must be at least 8")

    artifact = np.asarray(bundle["artifact"], dtype=np.float32)
    corrected = np.asarray(bundle["corrected"], dtype=np.float32)
    if artifact.shape != corrected.shape:
        raise ValueError(f"artifact and corrected arrays must have the same shape, got {artifact.shape} and {corrected.shape}")

    starts, stops = _epoch_boundaries(bundle)
    epoch_lengths = (stops - starts).astype(np.int64)
    clean_epochs = np.stack(
        [_resample_epoch(corrected[:, start:stop], target_epoch_samples) for start, stop in zip(starts, stops, strict=False)],
        axis=0,
    )
    artifact_epochs = np.stack(
        [_resample_epoch(artifact[:, start:stop], target_epoch_samples) for start, stop in zip(starts, stops, strict=False)],
        axis=0,
    )
    noisy_epochs = clean_epochs + artifact_epochs

    radius = context_epochs // 2
    if clean_epochs.shape[0] <= 2 * radius:
        raise ValueError("Not enough epochs for requested context width")

    clean_contexts: list[np.ndarray] = []
    artifact_contexts: list[np.ndarray] = []
    noisy_contexts: list[np.ndarray] = []
    length_contexts: list[np.ndarray] = []
    context_indices: list[np.ndarray] = []

    for center_idx in range(radius, clean_epochs.shape[0] - radius):
        idx = np.arange(center_idx - radius, center_idx + radius + 1, dtype=np.int64)
        clean_contexts.append(clean_epochs[idx])
        artifact_contexts.append(artifact_epochs[idx])
        noisy_contexts.append(noisy_epochs[idx])
        length_contexts.append(epoch_lengths[idx])
        context_indices.append(idx)

    clean_context = np.stack(clean_contexts, axis=0).astype(np.float32)
    artifact_context = np.stack(artifact_contexts, axis=0).astype(np.float32)
    noisy_context = np.stack(noisy_contexts, axis=0).astype(np.float32)
    artifact_epoch_lengths = np.stack(length_contexts, axis=0).astype(np.int64)
    artifact_context_indices = np.stack(context_indices, axis=0).astype(np.int64)
    center = radius

    n_examples = noisy_context.shape[0]
    source_id = np.full(n_examples, "niazy_aas_direct_artifact", dtype=object)
    source_path = np.full(n_examples, str(bundle.get("_source_path", "")), dtype=object)
    channel_indices = np.arange(artifact.shape[0], dtype=np.int64)

    phase = np.linspace(0.0, 1.0, target_epoch_samples, endpoint=False, dtype=np.float32)
    trigger_phase_linear = np.tile(phase, (n_examples, context_epochs, 1)).astype(np.float32)
    trigger_phase_sincos = np.tile(
        np.stack([np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)], axis=0),
        (n_examples, context_epochs, 1, 1),
    ).astype(np.float32)

    return {
        "noisy_context": noisy_context,
        "clean_context": clean_context,
        "artifact_context": artifact_context,
        "noisy_center": noisy_context[:, center],
        "clean_center": clean_context[:, center],
        "artifact_center": artifact_context[:, center],
        "artifact_epoch_lengths_samples": artifact_epoch_lengths,
        "artifact_context_indices": artifact_context_indices,
        "artifact_source_ids": source_id,
        "artifact_source_paths": source_path,
        "artifact_channel_indices": channel_indices,
        "trigger_phase_linear": trigger_phase_linear,
        "trigger_phase_sincos": trigger_phase_sincos,
        "sfreq": np.asarray([float(bundle["sfreq"][0])], dtype=np.float64),
        "ch_names": np.asarray(bundle["ch_names"], dtype=object),
    }


def build_dataset(
    *,
    artifact_bundle: Path,
    output_dir: Path,
    context_epochs: int,
    target_epoch_samples: int,
) -> tuple[Path, Path, dict[str, np.ndarray], dict[str, object]]:
    artifact_bundle = artifact_bundle.expanduser()
    if not artifact_bundle.exists():
        raise FileNotFoundError(artifact_bundle)

    with np.load(artifact_bundle, allow_pickle=True) as loaded:
        bundle = {key: loaded[key] for key in loaded.files}
    bundle["_source_path"] = np.asarray(str(artifact_bundle), dtype=object)

    dataset = _build_context_dataset(
        bundle,
        context_epochs=context_epochs,
        target_epoch_samples=target_epoch_samples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / DATASET_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    np.savez_compressed(dataset_path, **dataset)

    lengths = dataset["artifact_epoch_lengths_samples"]
    metadata: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "artifact_bundle": str(artifact_bundle),
        "mode": "niazy_only_proof_fit_context",
        "clean_target": "aas_corrected_niazy_surrogate",
        "artifact_target": "aas_estimated_niazy_artifact",
        "warning": (
            "This dataset intentionally uses Niazy-derived AAS corrected data and AAS artifact estimates. "
            "It is suitable for proof-of-fit tests, not for independent generalization claims."
        ),
        "n_examples": int(dataset["noisy_context"].shape[0]),
        "context_epochs": int(context_epochs),
        "epoch_samples": int(target_epoch_samples),
        "n_channels": int(dataset["noisy_context"].shape[2]),
        "sampling_frequency_hz": float(dataset["sfreq"][0]),
        "min_native_epoch_length_samples": int(np.min(lengths)),
        "median_native_epoch_length_samples": float(np.median(lengths)),
        "max_native_epoch_length_samples": int(np.max(lengths)),
        "mean_abs_clean_uv": float(np.mean(np.abs(dataset["clean_context"])) * 1e6),
        "mean_abs_artifact_uv": float(np.mean(np.abs(dataset["artifact_context"])) * 1e6),
        "mean_abs_noisy_uv": float(np.mean(np.abs(dataset["noisy_context"])) * 1e6),
        "max_abs_artifact_uv": float(np.max(np.abs(dataset["artifact_context"])) * 1e6),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dataset_path, metadata_path, dataset, metadata


def main() -> None:
    args = parse_args()
    dataset_path, metadata_path, dataset, metadata = build_dataset(
        artifact_bundle=args.artifact_bundle,
        output_dir=args.output_dir,
        context_epochs=args.context_epochs,
        target_epoch_samples=args.target_epoch_samples,
    )
    print("Saved Niazy-only proof-fit context dataset:")
    print(f"  dataset      : {dataset_path}")
    print(f"  metadata     : {metadata_path}")
    print(f"  examples     : {metadata['n_examples']}")
    print(f"  shape        : {dataset['noisy_context'].shape}")
    print(
        "  native length: "
        f"{metadata['min_native_epoch_length_samples']} / "
        f"{metadata['median_native_epoch_length_samples']:.1f} / "
        f"{metadata['max_native_epoch_length_samples']}"
    )
    print(f"  mean |clean| : {metadata['mean_abs_clean_uv']:.3f} uV")
    print(f"  mean |art|   : {metadata['mean_abs_artifact_uv']:.3f} uV")


if __name__ == "__main__":
    main()
