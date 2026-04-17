"""Extract the estimated fMRI artifact signal from the Niazy example dataset.

This script mirrors the existing FACETpy pipe-operator examples:

1. load the Niazy EEG-fMRI recording,
2. run a correction pipeline with ``|`` syntax,
3. persist both the corrected EEG and the estimated artifact signal.

The artifact export is meant as the first building block for a later synthetic
training dataset. The saved ``.npz`` bundle keeps the estimated artifact,
trigger positions, and core metadata together so it can be merged later with
clean spike EEG windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

from facet import (
    AASCorrection,
    CutAcquisitionWindow,
    DownSample,
    DropChannels,
    HighPassFilter,
    LowPassFilter,
    PasteAcquisitionWindow,
    PCACorrection,
    TriggerAligner,
    TriggerDetector,
    UpSample,
    load,
)

INPUT_FILE = Path("./examples/datasets/NiazyFMRI.edf")
OUTPUT_DIR = Path("./output/niazy_artifact_extraction")
TRIGGER_REGEX = r"\b1\b"
UPSAMPLE = 10
NON_EEG_CHANNELS = ["EKG", "EMG", "EOG", "ECG"]

CORRECTED_FIF = OUTPUT_DIR / "niazy_corrected_raw.fif"
ARTIFACT_FIF = OUTPUT_DIR / "niazy_estimated_artifact_raw.fif"
ARTIFACT_NPZ = OUTPUT_DIR / "niazy_estimated_artifact.npz"
METADATA_JSON = OUTPUT_DIR / "niazy_estimated_artifact_metadata.json"


def _artifact_raw_from_context(context) -> mne.io.BaseRaw:
    """Build an MNE RawArray from the accumulated FACETpy noise estimate."""
    estimated_noise = context.get_estimated_noise()
    if estimated_noise is None:
        raise RuntimeError("No estimated artifact signal found. The correction pipeline did not populate noise.")

    raw = context.get_raw()
    artifact_info = raw.info.copy()
    artifact_raw = mne.io.RawArray(estimated_noise.astype(np.float32), artifact_info, verbose=False)
    artifact_raw.set_meas_date(raw.info["meas_date"])
    return artifact_raw


def _write_artifact_bundle(context, artifact_raw: mne.io.BaseRaw) -> None:
    """Persist the estimated artifact both as MNE Raw and as a compact NumPy bundle."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corrected_raw = context.get_raw().copy()
    corrected_raw.save(CORRECTED_FIF, overwrite=True, verbose=False)
    artifact_raw.save(ARTIFACT_FIF, overwrite=True, verbose=False)

    triggers = context.get_triggers()
    artifact_length = context.get_artifact_length()
    acquisition = context.metadata.custom.get("acquisition", {})
    history = [step.name for step in context.get_history()]

    np.savez_compressed(
        ARTIFACT_NPZ,
        artifact=artifact_raw.get_data().astype(np.float32, copy=False),
        corrected=corrected_raw.get_data().astype(np.float32, copy=False),
        ch_names=np.asarray(artifact_raw.ch_names, dtype=object),
        sfreq=np.asarray([artifact_raw.info["sfreq"]], dtype=np.float64),
        triggers=np.asarray(triggers if triggers is not None else [], dtype=np.int64),
        artifact_length=np.asarray([-1 if artifact_length is None else int(artifact_length)], dtype=np.int64),
        artifact_to_trigger_offset=np.asarray([context.metadata.artifact_to_trigger_offset], dtype=np.float64),
        acq_start_sample=np.asarray([acquisition.get("acq_start_sample", -1)], dtype=np.int64),
        acq_end_sample=np.asarray([acquisition.get("acq_end_sample", -1)], dtype=np.int64),
        pre_trigger_samples=np.asarray([acquisition.get("pre_trigger_samples", -1)], dtype=np.int64),
        post_trigger_samples=np.asarray([acquisition.get("post_trigger_samples", -1)], dtype=np.int64),
    )

    metadata = {
        "input_file": str(INPUT_FILE),
        "corrected_fif": str(CORRECTED_FIF),
        "artifact_fif": str(ARTIFACT_FIF),
        "artifact_npz": str(ARTIFACT_NPZ),
        "n_channels": len(artifact_raw.ch_names),
        "n_samples": int(artifact_raw.n_times),
        "sfreq": float(artifact_raw.info["sfreq"]),
        "trigger_count": 0 if triggers is None else int(len(triggers)),
        "artifact_length_samples": None if artifact_length is None else int(artifact_length),
        "artifact_to_trigger_offset_seconds": float(context.metadata.artifact_to_trigger_offset),
        "acquisition": acquisition,
        "history": history,
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    context = load(str(INPUT_FILE), preload=True, artifact_to_trigger_offset=-0.005)

    context = (
        context
        | DropChannels(channels=NON_EEG_CHANNELS)
        | TriggerDetector(regex=TRIGGER_REGEX)
        | CutAcquisitionWindow()
        | HighPassFilter(freq=1.0)
        | UpSample(factor=UPSAMPLE)
        | TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False)
        | AASCorrection(window_size=30, correlation_threshold=0.975)
        | PCACorrection(n_components=0.95, hp_freq=1.0)
        | DownSample(factor=UPSAMPLE)
        | PasteAcquisitionWindow()
        | LowPassFilter(freq=70.0)
    )

    artifact_raw = _artifact_raw_from_context(context)
    _write_artifact_bundle(context, artifact_raw)

    print("Saved corrected EEG and estimated artifact bundle:")
    print(f"  corrected raw: {CORRECTED_FIF}")
    print(f"  artifact raw : {ARTIFACT_FIF}")
    print(f"  artifact npz : {ARTIFACT_NPZ}")
    print(f"  metadata json: {METADATA_JSON}")
    print(f"  channels     : {len(artifact_raw.ch_names)}")
    print(f"  samples      : {artifact_raw.n_times}")
    print(f"  sfreq        : {artifact_raw.info['sfreq']} Hz")


if __name__ == "__main__":
    main()
