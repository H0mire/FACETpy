"""Build a synthetic epoch-context artifact dataset from multiple clean sources.

This is the generic replacement for one-off spike dataset builders. It accepts
any number of clean EEG recordings, each with its own annotation regex, plus one
or more artifact libraries. The output keeps the same NPZ contract used by the
current deep-learning model packages:

- ``noisy_context``: ``(examples, context_epochs, 1, target_epoch_samples)``
- ``clean_context``: ``(examples, context_epochs, 1, target_epoch_samples)``
- ``artifact_context``: ``(examples, context_epochs, 1, target_epoch_samples)``
- ``artifact_center``: ``(examples, 1, target_epoch_samples)``

Example:

    uv run python examples/build_epoch_context_dataset.py \
        --clean-source output/synthetic_spike_source/synthetic_spike_source_raw.fif::spike_onset \
        --artifact-library output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
        --artifact-library output/artifact_libraries/large_mff_aas/large_mff_aas_artifact.npz \
        --target-epoch-samples 512 \
        --context-epochs 7 \
        --output-dir output/epoch_context_dataset_512
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_OUTPUT_DIR = Path("./output/epoch_context_dataset")
DEFAULT_CONTEXT_EPOCHS = 7
DEFAULT_ARTIFACT_SCALE_RANGE = (0.75, 1.25)
DEFAULT_MAX_JITTER_SAMPLES = 8
DEFAULT_SEED = 11
DEFAULT_ONSET_MIN_RATIO = 0.15
DEFAULT_ONSET_MAX_RATIO = 0.75
DATASET_FILENAME = "synthetic_spike_artifact_context_dataset.npz"
METADATA_FILENAME = "synthetic_spike_artifact_context_dataset_metadata.json"


@dataclass(frozen=True)
class CleanSourceSpec:
    path: Path
    annotation_regex: str


def _load_legacy_builder() -> Any:
    """Load shared artifact/context helpers from the existing builder script."""
    module_path = Path(__file__).with_name("build_synthetic_spike_artifact_context_dataset.py")
    spec = importlib.util.spec_from_file_location("_facet_epoch_context_legacy_builder", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {module_path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_clean_source(value: str) -> CleanSourceSpec:
    if "::" not in value:
        raise argparse.ArgumentTypeError(
            "clean sources must use PATH::ANNOTATION_REGEX, for example data.fif::spike_onset"
        )
    path_text, regex = value.split("::", 1)
    path = Path(path_text).expanduser()
    if not path_text.strip():
        raise argparse.ArgumentTypeError("clean source path must not be empty")
    if not regex.strip():
        raise argparse.ArgumentTypeError("clean source annotation regex must not be empty")
    return CleanSourceSpec(path=path, annotation_regex=regex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-source",
        type=_parse_clean_source,
        action="append",
        required=True,
        help="Clean EEG source in PATH::ANNOTATION_REGEX format. May be provided multiple times.",
    )
    parser.add_argument(
        "--artifact-library",
        type=Path,
        action="append",
        required=True,
        help="Artifact library .npz file or directory containing .npz bundles. May be provided multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-epochs", type=int, default=DEFAULT_CONTEXT_EPOCHS)
    parser.add_argument(
        "--target-epoch-samples",
        type=int,
        required=True,
        help="Final resampled artifact epoch length used by the model, e.g. 512.",
    )
    parser.add_argument("--artifact-scale-min", type=float, default=DEFAULT_ARTIFACT_SCALE_RANGE[0])
    parser.add_argument("--artifact-scale-max", type=float, default=DEFAULT_ARTIFACT_SCALE_RANGE[1])
    parser.add_argument("--max-jitter-samples", type=int, default=DEFAULT_MAX_JITTER_SAMPLES)
    parser.add_argument("--onset-min-ratio", type=float, default=DEFAULT_ONSET_MIN_RATIO)
    parser.add_argument("--onset-max-ratio", type=float, default=DEFAULT_ONSET_MAX_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _merge_dataset_chunks(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("No dataset chunks were produced from clean sources")
    keys = chunks[0].keys()
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def _aggregate_stats(
    dataset: dict[str, np.ndarray],
    *,
    context_epochs: int,
    target_epoch_samples: int,
    n_clean_sources: int,
    n_artifact_sources: int,
) -> dict[str, int | float]:
    lengths = dataset["artifact_epoch_lengths_samples"]
    return {
        "n_examples": int(dataset["noisy_context"].shape[0]),
        "context_epochs": int(context_epochs),
        "epoch_samples": int(target_epoch_samples),
        "mean_abs_clean": float(np.mean(np.abs(dataset["clean_context"]))),
        "mean_abs_artifact": float(np.mean(np.abs(dataset["artifact_context"]))),
        "mean_abs_noisy": float(np.mean(np.abs(dataset["noisy_context"]))),
        "min_artifact_epoch_length_samples": int(np.min(lengths)),
        "max_artifact_epoch_length_samples": int(np.max(lengths)),
        "median_artifact_epoch_length_samples": float(np.median(lengths)),
        "n_clean_sources": int(n_clean_sources),
        "n_artifact_sources": int(n_artifact_sources),
    }


def _write_dataset(
    *,
    output_dir: Path,
    clean_sources: list[CleanSourceSpec],
    artifact_sources: list[Any],
    dataset: dict[str, np.ndarray],
    target_sfreq: float,
    stats: dict[str, int | float],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / DATASET_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    np.savez_compressed(dataset_path, sfreq=np.asarray([target_sfreq], dtype=np.float64), **dataset)

    metadata = {
        "clean_sources": [
            {
                "path": str(source.path),
                "annotation_regex": source.annotation_regex,
            }
            for source in clean_sources
        ],
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
        "sampling_frequency_hz": float(target_sfreq),
        "mode": "multi_clean_source_single_channel_context",
        "center_target": "artifact",
        **stats,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dataset_path, metadata_path


def build_dataset_from_args(args: argparse.Namespace) -> tuple[Path, Path, dict[str, np.ndarray], dict[str, int | float]]:
    if args.context_epochs < 3 or args.context_epochs % 2 == 0:
        raise ValueError("--context-epochs must be an odd integer >= 3")
    if args.target_epoch_samples < 8:
        raise ValueError("--target-epoch-samples must be at least 8")

    helper = _load_legacy_builder()
    rng = np.random.default_rng(args.seed)

    artifact_bundle_paths = helper._discover_artifact_bundle_paths(args.artifact_library, args.artifact_library[0])
    first_bundle = helper._load_artifact_bundle(artifact_bundle_paths[0])
    target_sfreq = float(first_bundle["sfreq"][0])

    artifact_sources = helper._build_artifact_library_sources(
        artifact_bundle_paths,
        target_epoch_samples=args.target_epoch_samples,
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
    ) = helper._concatenate_artifact_sources(artifact_sources)

    dataset_chunks: list[dict[str, np.ndarray]] = []
    for source_index, clean_source in enumerate(args.clean_source):
        clean_raw = helper._load_clean_raw(clean_source.path)
        clean_raw = helper._resample_if_needed(clean_raw, target_sfreq)
        clean_contexts, onset_offsets, descriptions = helper._extract_clean_contexts(
            clean_raw,
            annotation_regex=clean_source.annotation_regex,
            context_epochs=args.context_epochs,
            epoch_samples=args.target_epoch_samples,
            onset_min_ratio=args.onset_min_ratio,
            onset_max_ratio=args.onset_max_ratio,
            rng=rng,
        )
        dataset_chunk, _stats = helper._build_single_channel_context_dataset(
            clean_contexts=clean_contexts,
            clean_channel_names=clean_raw.ch_names,
            onset_offsets=onset_offsets,
            spike_descriptions=descriptions,
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
        n_examples = int(dataset_chunk["noisy_context"].shape[0])
        dataset_chunk["clean_source_indices"] = np.full(n_examples, source_index, dtype=np.int64)
        dataset_chunk["clean_source_paths"] = np.full(n_examples, str(clean_source.path), dtype=object)
        dataset_chunk["clean_annotation_regexes"] = np.full(n_examples, clean_source.annotation_regex, dtype=object)
        dataset_chunks.append(dataset_chunk)

    dataset = _merge_dataset_chunks(dataset_chunks)
    stats = _aggregate_stats(
        dataset,
        context_epochs=args.context_epochs,
        target_epoch_samples=args.target_epoch_samples,
        n_clean_sources=len(args.clean_source),
        n_artifact_sources=len(artifact_sources),
    )
    dataset_path, metadata_path = _write_dataset(
        output_dir=args.output_dir,
        clean_sources=args.clean_source,
        artifact_sources=artifact_sources,
        dataset=dataset,
        target_sfreq=target_sfreq,
        stats=stats,
    )
    return dataset_path, metadata_path, dataset, stats


def main() -> None:
    dataset_path, metadata_path, dataset, stats = build_dataset_from_args(parse_args())
    print("Saved epoch-context artifact dataset:")
    print(f"  dataset       : {dataset_path}")
    print(f"  metadata      : {metadata_path}")
    print(f"  examples      : {stats['n_examples']}")
    print(f"  clean sources : {stats['n_clean_sources']}")
    print(f"  artifact srcs : {stats['n_artifact_sources']}")
    print(f"  noisy         : {dataset['noisy_context'].shape}")
    print(f"  center        : {dataset['artifact_center'].shape}")
    print(
        "  epoch lengths : "
        f"{stats['min_artifact_epoch_length_samples']} / "
        f"{stats['median_artifact_epoch_length_samples']:.1f} / "
        f"{stats['max_artifact_epoch_length_samples']}"
    )


if __name__ == "__main__":
    main()
