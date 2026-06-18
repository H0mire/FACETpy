"""CLI for the Run 3 / Weg A spatio-temporal reference dataset.

Thin wrapper around
:func:`facet.training.spatiotemporal_builder.build_spatiotemporal_reference_dataset`.
Reads an artifact bundle (ideally ``niazy_aas_pca4_direct`` from
``extract_niazy_aas_pca4_artifact.py``), builds the ``(N, context_epochs, 1+k,
core+2*guard)`` decoupled dataset and writes it as ``.npz`` plus a metadata JSON.

Examples::

    # Decoupled (synthetic independent clean) — the run_3 default
    uv run python tools/dataset_building/build_spatiotemporal_reference_dataset.py \
        --artifact-bundle output/artifact_libraries/niazy_aas_pca4_direct/niazy_aas_pca4_artifact.npz \
        --clean-source synthetic \
        --output-dir output/weg_a_spatiotemporal_512

    # Spike-preservation foundation for run_6 (synthetic clean + known spikes)
    uv run python tools/dataset_building/build_spatiotemporal_reference_dataset.py \
        --artifact-bundle output/artifact_libraries/niazy_aas_pca4_direct/niazy_aas_pca4_artifact.npz \
        --clean-source synthetic --inject-spikes \
        --output-dir output/weg_a_spatiotemporal_spikes_512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from facet.training.spatiotemporal_builder import (
    build_spatiotemporal_reference_dataset,
    summarize,
)

DATASET_FILENAME = "weg_a_spatiotemporal_dataset.npz"
METADATA_FILENAME = "weg_a_spatiotemporal_dataset_metadata.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact-bundle", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--context-epochs", type=int, default=7)
    p.add_argument("--core-samples", type=int, default=512)
    p.add_argument("--guard-samples", type=int, default=32)
    p.add_argument("--k-neighbors", type=int, default=2)
    p.add_argument(
        "--clean-source",
        choices=["synthetic", "external", "aas_corrected"],
        default="synthetic",
    )
    p.add_argument("--external-clean", type=Path, default=None, help="NPZ with key 'clean' (n_channels, n_samples)")
    p.add_argument("--inject-spikes", action="store_true", help="run_6 spike-preservation foundation")
    p.add_argument("--spike-rate-hz", type=float, default=0.7)
    p.add_argument("--spike-amplitude-uv", type=float, default=40.0)
    p.add_argument("--spike-width-ms", type=float, default=20.0)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle_path = args.artifact_bundle.expanduser()
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)

    with np.load(bundle_path, allow_pickle=True) as loaded:
        bundle = {key: loaded[key] for key in loaded.files}

    external_clean = None
    if args.clean_source == "external":
        if args.external_clean is None:
            raise SystemExit("--clean-source external requires --external-clean PATH")
        with np.load(args.external_clean.expanduser(), allow_pickle=True) as ext:
            external_clean = ext["clean"]

    dataset = build_spatiotemporal_reference_dataset(
        bundle,
        context_epochs=args.context_epochs,
        core_samples=args.core_samples,
        guard_samples=args.guard_samples,
        k_neighbors=args.k_neighbors,
        clean_source=args.clean_source,
        external_clean=external_clean,
        inject_spikes_mode=args.inject_spikes,
        spike_rate_hz=args.spike_rate_hz,
        spike_amplitude_uv=args.spike_amplitude_uv,
        spike_width_ms=args.spike_width_ms,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / DATASET_FILENAME
    metadata_path = args.output_dir / METADATA_FILENAME
    np.savez_compressed(dataset_path, **dataset)

    summary = summarize(dataset)
    metadata = {
        "dataset_path": str(dataset_path),
        "artifact_bundle": str(bundle_path),
        "mode": "weg_a_decoupled_spatiotemporal",
        "decoupling": {
            "lever_1_artifact": "AAS + PCA/OBS(n_components=4, hp_freq=300) supplied by the bundle",
            "lever_2_clean": f"clean_source={args.clean_source}",
        },
        "warning": (
            "Run 3 / Weg A reference dataset. clean_source='synthetic'|'external' decouples "
            "the target from AAS; 'aas_corrected' is the old coupled baseline only. Full "
            "bandwidth, no 70 Hz low-pass (run_3 §2)."
        ),
        **summary,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved Weg A spatio-temporal reference dataset:")
    print(f"  dataset      : {dataset_path}")
    print(f"  metadata     : {metadata_path}")
    print(f"  examples     : {summary['n_examples']}")
    print(f"  input shape  : {tuple(summary['input_shape'])}  (epochs, 1+k, core+2*guard)")
    print(f"  target shape : {tuple(summary['target_shape'])}")
    print(f"  clean source : {summary['clean_source']}  (spikes injected: {summary['spikes_injected']})")
    print(f"  mean |clean| : {summary['mean_abs_clean_uv']:.3f} uV")
    print(f"  mean |art|   : {summary['mean_abs_artifact_uv']:.3f} uV")
    if summary["spikes_injected"]:
        print(f"  spike labels : {summary['spike_label_positive_fraction'] * 100:.3f}% positive samples")


if __name__ == "__main__":
    main()
