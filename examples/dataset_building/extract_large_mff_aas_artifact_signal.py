"""Extract early AAS artifact epochs from the large EGI MFF example dataset.

The large recording contains many channels and many fMRI trigger epochs. This
script deliberately loads only the first trigger window needed for an artifact
library and runs AAS at native sampling rate. No pre-AAS upsampling is applied.

The resulting bundle is compatible with
``examples/dataset_building/build_synthetic_spike_artifact_context_dataset.py``.

Example:
    uv run python examples/dataset_building/extract_large_mff_aas_artifact_signal.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

from facet import AASCorrection, DropChannels, HighPassFilter, TriggerAligner, TriggerDetector, load
from facet.preprocessing import RawTransform

INPUT_FILE = Path("/Volumes/JanikProSSD/DataSets/EEG Datasets/EEGfMRI_20250519_20180312_004257.mff")
OUTPUT_DIR = Path("./output/large_mff_artifact_extraction")
LIBRARY_DIR = Path("./output/artifact_libraries/large_mff_aas")
TRIGGER_REGEX = r"^1$"
STIM_CHANNEL = "TREV"
NON_EEG_CHANNELS = ["ECG", "EKG", "EMG", "EOG"]
DEFAULT_N_TRIGGERS = 96
DEFAULT_WINDOW_SIZE = 30

ARTIFACT_FIF = OUTPUT_DIR / "large_mff_aas_artifact_raw.fif"
ARTIFACT_NPZ = OUTPUT_DIR / "large_mff_aas_artifact.npz"
METADATA_JSON = OUTPUT_DIR / "large_mff_aas_artifact_metadata.json"
LIBRARY_NPZ = LIBRARY_DIR / "large_mff_aas_artifact.npz"
LIBRARY_METADATA_JSON = LIBRARY_DIR / "large_mff_aas_artifact_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--n-triggers", type=int, default=DEFAULT_N_TRIGGERS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--library-dir", type=Path, default=LIBRARY_DIR)
    return parser.parse_args()


def _discover_trigger_window(path: Path, n_triggers: int) -> tuple[int, int, int, int, float]:
    if n_triggers < 8:
        raise ValueError("n-triggers must be at least 8 so context windows can be built later")

    context = load(str(path), preload=False)
    raw = context.get_raw()
    events = mne.find_events(raw, stim_channel=STIM_CHANNEL, initial_event=True, verbose=False)
    events = events[events[:, 2] == 1]
    if len(events) < n_triggers:
        raise ValueError(f"Requested {n_triggers} triggers, but only found {len(events)}")

    selected = events[:n_triggers, 0].astype(np.int64, copy=False)
    trigger_diffs = np.diff(selected)
    artifact_length = int(np.median(trigger_diffs))
    start_sample = max(0, int(selected[0]))
    stop_sample = min(raw.n_times, int(selected[-1] + artifact_length))
    return start_sample, stop_sample, int(len(events)), artifact_length, float(raw.info["sfreq"])


def _pick_eeg_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels found for artifact export")
    return raw.copy().pick(eeg_picks, verbose=False)


def _artifact_raw_from_context(context) -> mne.io.BaseRaw:
    estimated_noise = context.get_estimated_noise()
    if estimated_noise is None:
        raise RuntimeError("No AAS artifact estimate found")

    raw = context.get_raw()
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    artifact_info = raw.copy().pick(eeg_picks, verbose=False).info.copy()
    artifact_raw = mne.io.RawArray(estimated_noise[eeg_picks].astype(np.float32), artifact_info, verbose=False)
    artifact_raw.set_meas_date(raw.info["meas_date"])
    return artifact_raw


def _write_bundle(
    *,
    output_dir: Path,
    library_dir: Path,
    input_file: Path,
    context,
    artifact_raw: mne.io.BaseRaw,
    total_trigger_count: int,
    start_sample: int,
    stop_sample: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    library_dir.mkdir(parents=True, exist_ok=True)

    artifact_raw.save(ARTIFACT_FIF, overwrite=True, verbose=False)
    triggers = np.asarray(context.get_triggers(), dtype=np.int64)
    artifact_length = context.get_artifact_length()
    corrected_raw = _pick_eeg_raw(context.get_raw())

    np.savez_compressed(
        ARTIFACT_NPZ,
        artifact=artifact_raw.get_data().astype(np.float32, copy=False),
        corrected=corrected_raw.get_data().astype(np.float32, copy=False),
        ch_names=np.asarray(artifact_raw.ch_names, dtype=object),
        sfreq=np.asarray([artifact_raw.info["sfreq"]], dtype=np.float64),
        triggers=triggers,
        artifact_length=np.asarray([-1 if artifact_length is None else int(artifact_length)], dtype=np.int64),
        artifact_to_trigger_offset=np.asarray([context.metadata.artifact_to_trigger_offset], dtype=np.float64),
        acq_start_sample=np.asarray([start_sample], dtype=np.int64),
        acq_end_sample=np.asarray([stop_sample], dtype=np.int64),
        pre_trigger_samples=np.asarray([-1], dtype=np.int64),
        post_trigger_samples=np.asarray([-1], dtype=np.int64),
    )

    metadata = {
        "input_file": str(input_file),
        "artifact_fif": str(ARTIFACT_FIF),
        "artifact_npz": str(ARTIFACT_NPZ),
        "library_npz": str(LIBRARY_NPZ),
        "source_type": "large_mff_aas_artifact_native_no_upsampling",
        "n_channels": len(artifact_raw.ch_names),
        "n_samples": int(artifact_raw.n_times),
        "sfreq": float(artifact_raw.info["sfreq"]),
        "trigger_count_used": int(len(triggers)),
        "trigger_count_total": int(total_trigger_count),
        "artifact_length_samples": None if artifact_length is None else int(artifact_length),
        "artifact_to_trigger_offset_seconds": float(context.metadata.artifact_to_trigger_offset),
        "sample_window": {
            "start_sample_original": int(start_sample),
            "stop_sample_original": int(stop_sample),
        },
        "history": [step.name for step in context.get_history()],
        "note": "AAS was run at native sampling rate with UPSAMPLE=1; only the first trigger window was loaded.",
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LIBRARY_NPZ.write_bytes(ARTIFACT_NPZ.read_bytes())
    LIBRARY_METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start_sample, stop_sample, total_trigger_count, estimated_artifact_length, sfreq = _discover_trigger_window(
        args.input_file,
        n_triggers=args.n_triggers,
    )

    context = load(
        str(args.input_file),
        preload=False,
        start_sample=start_sample,
        stop_sample=stop_sample,
        upsampling_factor=1,
    )
    context = (
        context
        | RawTransform("load_cropped_data", lambda raw: raw.copy().load_data(verbose=False))
        | DropChannels(channels=NON_EEG_CHANNELS)
        | TriggerDetector(regex=TRIGGER_REGEX)
        | HighPassFilter(freq=1.0)
        | TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False)
        | AASCorrection(window_size=args.window_size, correlation_threshold=0.975, realign_after_averaging=True)
    )

    artifact_raw = _artifact_raw_from_context(context)
    _write_bundle(
        output_dir=args.output_dir,
        library_dir=args.library_dir,
        input_file=args.input_file,
        context=context,
        artifact_raw=artifact_raw,
        total_trigger_count=total_trigger_count,
        start_sample=start_sample,
        stop_sample=stop_sample,
    )

    print("Saved large MFF AAS artifact bundle:")
    print(f"  source window : {start_sample}:{stop_sample} at {sfreq:.1f} Hz")
    print(f"  estimated len : {estimated_artifact_length} samples")
    print(f"  artifact raw  : {ARTIFACT_FIF}")
    print(f"  artifact npz  : {ARTIFACT_NPZ}")
    print(f"  library npz   : {LIBRARY_NPZ}")
    print(f"  metadata      : {METADATA_JSON}")
    print(f"  channels      : {len(artifact_raw.ch_names)}")
    print(f"  triggers used : {len(context.get_triggers())} / {total_trigger_count}")
    print(f"  samples       : {artifact_raw.n_times}")
    print(f"  sfreq         : {artifact_raw.info['sfreq']} Hz")


if __name__ == "__main__":
    main()
