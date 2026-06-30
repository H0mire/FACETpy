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


def extract_pretrigger_clean(
    edf_path: Path,
    *,
    trigger_regex: str = r"\b1\b",
    skip_s: float = 1.0,
    guard_s: float = 1.0,
) -> tuple[np.ndarray, float, list[str]]:
    """Extract the real GA-free pre-trigger clean (brain + BCG) from a Niazy EDF.

    Loads the recording, drops non-EEG channels, detects triggers, high-passes at
    1 Hz, and returns the EEG segment from ``skip_s`` to ``guard_s`` before the
    first trigger — in-scanner EEG with no gradient artifact yet but with the
    ballistocardiogram already present (run_3 §3 / cascade GA-model clean).
    """
    import mne  # noqa: PLC0415

    from facet import DropChannels, HighPassFilter, TriggerDetector, load  # noqa: PLC0415

    ctx = load(str(edf_path), preload=True, artifact_to_trigger_offset=-0.005)
    ctx = ctx | DropChannels(channels=["EKG", "EMG", "EOG", "ECG"]) | TriggerDetector(regex=trigger_regex) | HighPassFilter(freq=1.0)
    raw = ctx.get_raw()
    sfreq = float(raw.info["sfreq"])
    first_trigger = int(np.min(ctx.get_triggers()))
    picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude="bads")
    names = [raw.ch_names[i] for i in picks]
    start = int(skip_s * sfreq)
    stop = first_trigger - int(guard_s * sfreq)
    if stop - start < int(sfreq):
        raise SystemExit(f"Pre-trigger segment too short ({(stop - start) / sfreq:.1f}s); first trigger at {first_trigger / sfreq:.1f}s")
    clean = raw._data[picks, start:stop].astype(np.float32)
    return clean, sfreq, names


# Authoritative 29-channel order of the VEPISET IED dataset (github.com/vepiset/vepiset_dataset);
# rows 0-18 are the 19 standard 10-20 EEG channels.
VEPISET_EEG19 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]


def extract_real_ied_pool(
    dataset_dir: Path,
    *,
    max_ieds: int = 300,
    window_ms: float = 600.0,
    hp_freq: float = 1.0,
    sfreq: float = 500.0,
) -> tuple[list[dict], float]:
    """Build a pool of REAL annotated IEDs from the VEPISET dataset.

    Scans the ``.mat`` recordings for ``!`` spike onsets, extracts a multi-channel
    window over the 19 standard 10-20 EEG channels (rows 0-18), 1 Hz high-passed,
    and normalises each to its focal peak. The real morphology *and* the real
    cross-channel topography are preserved; channels are mapped to the target
    montage by name at injection time (run_3 §6.6 / run_6 ground truth).
    """
    import scipy.io as sio  # noqa: PLC0415
    from scipy.signal import butter, sosfiltfilt  # noqa: PLC0415

    mat_dir = dataset_dir / "MAT_Files"
    mats = sorted(mat_dir.glob("*.mat"))
    if not mats:
        raise SystemExit(f"No .mat files under {mat_dir}")
    sos = butter(4, hp_freq, btype="high", fs=sfreq, output="sos")
    half = int(0.5 * window_ms * 1e-3 * sfreq)
    pool: list[dict] = []
    for f in mats:
        m = sio.loadmat(str(f), squeeze_me=True, struct_as_record=False)
        if "events" not in m or "eeg_data" not in m:
            continue
        ev = np.asarray(m["events"]).reshape(-1, 3)
        onsets = [float(str(r[0]).strip()) for r in ev if str(r[2]).strip() == "!"]
        if not onsets:
            continue
        eeg = np.asarray(m["eeg_data"], dtype=np.float64)[: len(VEPISET_EEG19)]
        eeg = sosfiltfilt(sos, eeg, axis=1)
        for on in onsets:
            s = int(on * sfreq)
            a, b = s - half, s + half
            if a < 0 or b > eeg.shape[1]:
                continue
            win = eeg[:, a:b]
            focal = float(np.max(np.abs(win)))
            if focal <= 0:
                continue
            pool.append({"waveforms": (win / focal).astype(np.float32), "names": list(VEPISET_EEG19)})
            if len(pool) >= max_ieds:
                break
        if len(pool) >= max_ieds:
            break
    if not pool:
        raise SystemExit("No IEDs with '!' markers found in the dataset")
    return pool, sfreq


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
        choices=["synthetic", "external", "aas_corrected", "niazy_pretrigger"],
        default="synthetic",
    )
    p.add_argument("--external-clean", type=Path, default=None, help="NPZ with key 'clean' (n_channels, n_samples)")
    p.add_argument(
        "--niazy-edf",
        type=Path,
        default=Path("./examples/datasets/NiazyFMRI.edf"),
        help="Niazy EDF for clean_source=niazy_pretrigger (real brain+BCG, GA-free pre-trigger segment)",
    )
    p.add_argument("--pretrigger-guard-s", type=float, default=1.0, help="Stop the clean segment this many s before the first trigger")
    p.add_argument("--pretrigger-skip-s", type=float, default=1.0, help="Skip this many s at the start (filter edge transient)")
    p.add_argument("--inject-spikes", action="store_true", help="run_6 spike-preservation foundation")
    p.add_argument("--spike-source", choices=["synthetic", "real_ied"], default="synthetic")
    p.add_argument(
        "--ied-dataset",
        type=Path,
        default=Path("/Volumes/JanikProSSD/DataSets/opensource-dataset"),
        help="VEPISET IED dataset dir for --spike-source real_ied (real annotated spikes)",
    )
    p.add_argument("--max-ieds", type=int, default=300, help="how many real IEDs to load into the pool")
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

    pretrigger_clean = None
    pretrigger_sfreq = None
    if args.clean_source == "niazy_pretrigger":
        pretrigger_clean, pretrigger_sfreq, pre_names = extract_pretrigger_clean(
            args.niazy_edf.expanduser(), skip_s=args.pretrigger_skip_s, guard_s=args.pretrigger_guard_s
        )
        print(f"  pre-trigger clean: {pretrigger_clean.shape[0]} ch, {pretrigger_clean.shape[1] / pretrigger_sfreq:.1f}s @ {pretrigger_sfreq:.0f} Hz")

    real_ied_pool = None
    real_ied_sfreq = None
    if args.inject_spikes and args.spike_source == "real_ied":
        real_ied_pool, real_ied_sfreq = extract_real_ied_pool(args.ied_dataset.expanduser(), max_ieds=args.max_ieds)
        print(f"  real IED pool: {len(real_ied_pool)} annotated spikes @ {real_ied_sfreq:.0f} Hz")

    dataset = build_spatiotemporal_reference_dataset(
        bundle,
        context_epochs=args.context_epochs,
        core_samples=args.core_samples,
        guard_samples=args.guard_samples,
        k_neighbors=args.k_neighbors,
        clean_source=args.clean_source,
        external_clean=external_clean,
        pretrigger_clean=pretrigger_clean,
        pretrigger_sfreq=pretrigger_sfreq,
        inject_spikes_mode=args.inject_spikes,
        spike_source=args.spike_source,
        real_ied_pool=real_ied_pool,
        real_ied_sfreq=real_ied_sfreq,
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
