"""Extract fMRI artifact source signals from the Niazy example dataset.

This script mirrors the existing FACETpy pipe-operator examples:

1. load the Niazy EEG-fMRI recording,
2. run a trigger-detection and high-pass preprocessing pipeline with ``|`` syntax,
3. persist an EEG-only raw/high-pass gradient artifact proxy,
4. persist AAS noise estimates directly after AAS,
5. persist both the corrected EEG and the final low-pass estimated artifact signal.

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
    DownSample,
    DropChannels,
    HighPassFilter,
    LowPassFilter,
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
RAW_GRADIENT_FIF = OUTPUT_DIR / "niazy_raw_gradient_artifact_raw.fif"
RAW_GRADIENT_NPZ = OUTPUT_DIR / "niazy_raw_gradient_artifact.npz"
RAW_GRADIENT_METADATA_JSON = OUTPUT_DIR / "niazy_raw_gradient_artifact_metadata.json"
AAS_DIRECT_FIF = OUTPUT_DIR / "niazy_aas_direct_artifact_raw.fif"
AAS_DIRECT_NPZ = OUTPUT_DIR / "niazy_aas_direct_artifact.npz"
AAS_DIRECT_METADATA_JSON = OUTPUT_DIR / "niazy_aas_direct_artifact_metadata.json"
AAS_DOWNSAMPLED_FIF = OUTPUT_DIR / "niazy_aas_downsampled_artifact_raw.fif"
AAS_DOWNSAMPLED_NPZ = OUTPUT_DIR / "niazy_aas_downsampled_artifact.npz"
AAS_DOWNSAMPLED_METADATA_JSON = OUTPUT_DIR / "niazy_aas_downsampled_artifact_metadata.json"


def _pick_eeg_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels found for artifact export")
    return raw.copy().pick(eeg_picks, verbose=False)


def _artifact_raw_from_context(context) -> mne.io.BaseRaw:
    """Build an MNE RawArray from the accumulated FACETpy noise estimate."""
    estimated_noise = context.get_estimated_noise()
    if estimated_noise is None:
        raise RuntimeError("No estimated artifact signal found. The correction pipeline did not populate noise.")

    raw = context.get_raw()
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    artifact_info = raw.copy().pick(eeg_picks, verbose=False).info.copy()
    artifact_raw = mne.io.RawArray(estimated_noise[eeg_picks].astype(np.float32), artifact_info, verbose=False)
    artifact_raw.set_meas_date(raw.info["meas_date"])
    return artifact_raw


def _bundle_metadata(context) -> tuple[np.ndarray, int | None, dict[str, object], list[str]]:
    triggers = context.get_triggers()
    artifact_length = context.get_artifact_length()
    acquisition = context.metadata.custom.get("acquisition", {})
    history = [step.name for step in context.get_history()]
    return np.asarray(triggers if triggers is not None else [], dtype=np.int64), artifact_length, acquisition, history


def _write_npz_bundle(
    *,
    path: Path,
    context,
    artifact_raw: mne.io.BaseRaw,
    corrected_raw: mne.io.BaseRaw | None,
) -> None:
    triggers, artifact_length, acquisition, _history = _bundle_metadata(context)
    np.savez_compressed(
        path,
        artifact=artifact_raw.get_data().astype(np.float32, copy=False),
        corrected=(
            corrected_raw.get_data().astype(np.float32, copy=False)
            if corrected_raw is not None
            else np.zeros(artifact_raw.get_data().shape, dtype=np.float32)
        ),
        ch_names=np.asarray(artifact_raw.ch_names, dtype=object),
        sfreq=np.asarray([artifact_raw.info["sfreq"]], dtype=np.float64),
        triggers=triggers,
        artifact_length=np.asarray([-1 if artifact_length is None else int(artifact_length)], dtype=np.int64),
        artifact_to_trigger_offset=np.asarray([context.metadata.artifact_to_trigger_offset], dtype=np.float64),
        acq_start_sample=np.asarray([acquisition.get("acq_start_sample", -1)], dtype=np.int64),
        acq_end_sample=np.asarray([acquisition.get("acq_end_sample", -1)], dtype=np.int64),
        pre_trigger_samples=np.asarray([acquisition.get("pre_trigger_samples", -1)], dtype=np.int64),
        post_trigger_samples=np.asarray([acquisition.get("post_trigger_samples", -1)], dtype=np.int64),
    )


def _write_artifact_bundle(context, artifact_raw: mne.io.BaseRaw) -> None:
    """Persist the estimated artifact both as MNE Raw and as a compact NumPy bundle."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corrected_raw = _pick_eeg_raw(context.get_raw())
    corrected_raw.save(CORRECTED_FIF, overwrite=True, verbose=False)
    artifact_raw.save(ARTIFACT_FIF, overwrite=True, verbose=False)
    _write_npz_bundle(path=ARTIFACT_NPZ, context=context, artifact_raw=artifact_raw, corrected_raw=corrected_raw)

    triggers, artifact_length, acquisition, history = _bundle_metadata(context)
    metadata = {
        "input_file": str(INPUT_FILE),
        "corrected_fif": str(CORRECTED_FIF),
        "artifact_fif": str(ARTIFACT_FIF),
        "artifact_npz": str(ARTIFACT_NPZ),
        "source_type": "aas_estimated_artifact_eeg_only",
        "n_channels": len(artifact_raw.ch_names),
        "n_samples": int(artifact_raw.n_times),
        "sfreq": float(artifact_raw.info["sfreq"]),
        "trigger_count": int(len(triggers)),
        "artifact_length_samples": None if artifact_length is None else int(artifact_length),
        "artifact_to_trigger_offset_seconds": float(context.metadata.artifact_to_trigger_offset),
        "acquisition": acquisition,
        "history": history,
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_aas_artifact_bundle(
    *,
    context,
    artifact_raw: mne.io.BaseRaw,
    artifact_fif: Path,
    artifact_npz: Path,
    metadata_json: Path,
    source_type: str,
) -> None:
    """Persist an intermediate AAS noise estimate without final low-pass filtering."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corrected_raw = _pick_eeg_raw(context.get_raw())
    artifact_raw.save(artifact_fif, overwrite=True, verbose=False)
    _write_npz_bundle(path=artifact_npz, context=context, artifact_raw=artifact_raw, corrected_raw=corrected_raw)

    triggers, artifact_length, acquisition, history = _bundle_metadata(context)
    metadata = {
        "input_file": str(INPUT_FILE),
        "artifact_fif": str(artifact_fif),
        "artifact_npz": str(artifact_npz),
        "source_type": source_type,
        "n_channels": len(artifact_raw.ch_names),
        "n_samples": int(artifact_raw.n_times),
        "sfreq": float(artifact_raw.info["sfreq"]),
        "trigger_count": int(len(triggers)),
        "artifact_length_samples": None if artifact_length is None else int(artifact_length),
        "artifact_to_trigger_offset_seconds": float(context.metadata.artifact_to_trigger_offset),
        "acquisition": acquisition,
        "history": history,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_raw_gradient_bundle(context) -> mne.io.BaseRaw:
    """Persist the EEG-only high-pass raw signal as a gradient-artifact proxy."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_gradient = _pick_eeg_raw(context.get_raw())
    raw_gradient.save(RAW_GRADIENT_FIF, overwrite=True, verbose=False)
    _write_npz_bundle(path=RAW_GRADIENT_NPZ, context=context, artifact_raw=raw_gradient, corrected_raw=None)

    triggers, artifact_length, acquisition, history = _bundle_metadata(context)
    metadata = {
        "input_file": str(INPUT_FILE),
        "artifact_fif": str(RAW_GRADIENT_FIF),
        "artifact_npz": str(RAW_GRADIENT_NPZ),
        "source_type": "raw_highpass_niazy_gradient_proxy_eeg_only",
        "n_channels": len(raw_gradient.ch_names),
        "n_samples": int(raw_gradient.n_times),
        "sfreq": float(raw_gradient.info["sfreq"]),
        "trigger_count": int(len(triggers)),
        "artifact_length_samples": None if artifact_length is None else int(artifact_length),
        "artifact_to_trigger_offset_seconds": float(context.metadata.artifact_to_trigger_offset),
        "acquisition": acquisition,
        "history": history,
        "warning": (
            "This source is the EEG-only high-pass raw signal around Niazy fMRI triggers. "
            "It preserves the visually realistic gradient artifact morphology but can contain residual EEG."
        ),
    }
    RAW_GRADIENT_METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return raw_gradient


def main() -> None:
    context = load(str(INPUT_FILE), preload=True, artifact_to_trigger_offset=-0.005)

    context = (
        context
        | DropChannels(channels=NON_EEG_CHANNELS)
        | TriggerDetector(regex=TRIGGER_REGEX)
        | HighPassFilter(freq=1.0)
    )
    raw_gradient = _write_raw_gradient_bundle(context)

    context = (
        context
        | UpSample(factor=UPSAMPLE)
        | TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False)
        | AASCorrection(window_size=30, correlation_threshold=0.975)
    )

    aas_direct_artifact_raw = _artifact_raw_from_context(context)
    _write_aas_artifact_bundle(
        context=context,
        artifact_raw=aas_direct_artifact_raw,
        artifact_fif=AAS_DIRECT_FIF,
        artifact_npz=AAS_DIRECT_NPZ,
        metadata_json=AAS_DIRECT_METADATA_JSON,
        source_type="aas_direct_artifact_eeg_only_upsampled_no_lowpass",
    )

    context = context | DownSample(factor=UPSAMPLE)
    aas_downsampled_artifact_raw = _artifact_raw_from_context(context)
    _write_aas_artifact_bundle(
        context=context,
        artifact_raw=aas_downsampled_artifact_raw,
        artifact_fif=AAS_DOWNSAMPLED_FIF,
        artifact_npz=AAS_DOWNSAMPLED_NPZ,
        metadata_json=AAS_DOWNSAMPLED_METADATA_JSON,
        source_type="aas_artifact_eeg_only_downsampled_no_final_lowpass",
    )

    context = context | LowPassFilter(freq=70.0)
    artifact_raw = _artifact_raw_from_context(context)
    _write_artifact_bundle(context, artifact_raw)

    print("Saved Niazy artifact bundles:")
    print(f"  raw gradient raw : {RAW_GRADIENT_FIF}")
    print(f"  raw gradient npz : {RAW_GRADIENT_NPZ}")
    print(f"  raw gradient meta: {RAW_GRADIENT_METADATA_JSON}")
    print(f"  aas direct raw   : {AAS_DIRECT_FIF}")
    print(f"  aas direct npz   : {AAS_DIRECT_NPZ}")
    print(f"  aas direct meta  : {AAS_DIRECT_METADATA_JSON}")
    print(f"  aas down raw     : {AAS_DOWNSAMPLED_FIF}")
    print(f"  aas down npz     : {AAS_DOWNSAMPLED_NPZ}")
    print(f"  aas down meta    : {AAS_DOWNSAMPLED_METADATA_JSON}")
    print(f"  corrected raw: {CORRECTED_FIF}")
    print(f"  artifact raw : {ARTIFACT_FIF}")
    print(f"  artifact npz : {ARTIFACT_NPZ}")
    print(f"  metadata json: {METADATA_JSON}")
    print(
        "  channels     : "
        f"{len(raw_gradient.ch_names)} raw-gradient, "
        f"{len(aas_direct_artifact_raw.ch_names)} aas-direct, "
        f"{len(aas_downsampled_artifact_raw.ch_names)} aas-downsampled, "
        f"{len(artifact_raw.ch_names)} final-estimated"
    )
    print(f"  samples      : {artifact_raw.n_times}")
    print(f"  sfreq        : {artifact_raw.info['sfreq']} Hz")


if __name__ == "__main__":
    main()
