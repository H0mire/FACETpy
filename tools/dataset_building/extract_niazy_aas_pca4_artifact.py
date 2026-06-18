"""Extract the *combined* AAS + PCA/OBS artifact estimate (Run 3 / Weg A, Hebel 1).

This is the bundle producer for ``docs/research/run_3_decoupled_dataset_weg_a.md``
§2/§6.1. It mirrors ``examples/dataset_building/extract_niazy_artifact_signal.py``
but inserts one extra step after AAS:

    ... | AASCorrection(...) | PCACorrection(n_components=4, hp_freq=300.0)

``PCACorrection`` (FACET ``DoPCA``/``FitOBS``) adds its OBS reconstruction to the
context's accumulated noise, so ``context.get_estimated_noise()`` returns
**AAS + PCA(4)** — a more complete artifact template than AAS alone, capturing the
>300 Hz slice-to-slice residual AAS leaves behind (run_3 §2).

Deliberate choices (run_3 §2/§8):

* ``hp_freq=300`` keeps the EEG band out of the OBS basis, so the OBS models only
  the high-frequency residual and cannot remove brain signal.
* **No 70 Hz low-pass** and **no down-sample** — the artifact is exported at the
  upsampled, full Nyquist bandwidth so the model learns it in its entirety.

Output bundle (``niazy_aas_pca4_direct``) is drop-in for the AAS-only bundles: same
keys (``artifact``, ``corrected``, ``triggers``, ``sfreq``,
``artifact_to_trigger_offset``, ``ch_names``, acquisition window). Channel
positions for k-NN are derived downstream from ``ch_names`` via the standard_1005
montage, so they are not stored here.

Example::

    uv run python tools/dataset_building/extract_niazy_aas_pca4_artifact.py \
        --upsample-factor 10 \
        --output-dir output/artifact_libraries/niazy_aas_pca4_direct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

from facet import (
    AASCorrection,
    DropChannels,
    HighPassFilter,
    PCACorrection,
    TriggerAligner,
    TriggerDetector,
    UpSample,
    load,
)

DEFAULT_INPUT = Path("./examples/datasets/NiazyFMRI.edf")
DEFAULT_OUTPUT_DIR = Path("./output/artifact_libraries/niazy_aas_pca4_direct")
TRIGGER_REGEX = r"\b1\b"
NON_EEG_CHANNELS = ["EKG", "EMG", "EOG", "ECG"]

ARTIFACT_NPZ_NAME = "niazy_aas_pca4_artifact.npz"
METADATA_NAME = "niazy_aas_pca4_artifact_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--upsample-factor", type=int, default=10)
    parser.add_argument("--trigger-regex", type=str, default=TRIGGER_REGEX)
    parser.add_argument("--n-components", type=int, default=4, help="OBS/PCA components (Niazy classic = 4)")
    parser.add_argument("--hp-freq", type=float, default=300.0, help="OBS high-pass cutoff in Hz")
    parser.add_argument("--aas-window-size", type=int, default=30)
    parser.add_argument("--aas-correlation-threshold", type=float, default=0.975)
    return parser.parse_args()


def _pick_eeg(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    if len(picks) == 0:
        raise RuntimeError("No EEG channels found")
    return raw.copy().pick(picks, verbose=False)


def _artifact_raw(context) -> mne.io.BaseRaw:
    estimated = context.get_estimated_noise()
    if estimated is None:
        raise RuntimeError("No estimated artifact — the correction pipeline did not populate noise")
    raw = context.get_raw()
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    info = raw.copy().pick(picks, verbose=False).info.copy()
    art = mne.io.RawArray(estimated[picks].astype(np.float32), info, verbose=False)
    art.set_meas_date(raw.info["meas_date"])
    return art


def main() -> None:
    args = parse_args()
    if args.upsample_factor < 1:
        raise ValueError("--upsample-factor must be >= 1")

    context = load(str(args.input_file), preload=True, artifact_to_trigger_offset=-0.005)
    context = (
        context
        | DropChannels(channels=NON_EEG_CHANNELS)
        | TriggerDetector(regex=args.trigger_regex)
        | HighPassFilter(freq=1.0)
        | UpSample(factor=args.upsample_factor)
        | TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False)
        | AASCorrection(window_size=args.aas_window_size, correlation_threshold=args.aas_correlation_threshold)
        | PCACorrection(n_components=args.n_components, hp_freq=args.hp_freq)
    )

    artifact_raw = _artifact_raw(context)          # AAS + PCA combined estimate
    corrected_raw = _pick_eeg(context.get_raw())   # original - (AAS + PCA)

    triggers = context.get_triggers()
    triggers = np.asarray(triggers if triggers is not None else [], dtype=np.int64)
    artifact_length = context.get_artifact_length()
    acquisition = context.metadata.custom.get("acquisition", {})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / ARTIFACT_NPZ_NAME
    np.savez_compressed(
        npz_path,
        artifact=artifact_raw.get_data().astype(np.float32, copy=False),
        corrected=corrected_raw.get_data().astype(np.float32, copy=False),
        ch_names=np.asarray(artifact_raw.ch_names, dtype=object),
        sfreq=np.asarray([artifact_raw.info["sfreq"]], dtype=np.float64),
        triggers=triggers,
        artifact_length=np.asarray([-1 if artifact_length is None else int(artifact_length)], dtype=np.int64),
        artifact_to_trigger_offset=np.asarray([context.metadata.artifact_to_trigger_offset], dtype=np.float64),
        acq_start_sample=np.asarray([acquisition.get("acq_start_sample", -1)], dtype=np.int64),
        acq_end_sample=np.asarray([acquisition.get("acq_end_sample", -1)], dtype=np.int64),
    )

    metadata = {
        "input_file": str(args.input_file),
        "artifact_npz": str(npz_path),
        "source_type": "aas_plus_pca_obs_artifact_eeg_only_upsampled_no_lowpass",
        "artifact_composition": f"AAS + PCACorrection(n_components={args.n_components}, hp_freq={args.hp_freq})",
        "n_components": int(args.n_components),
        "obs_hp_freq_hz": float(args.hp_freq),
        "upsample_factor": int(args.upsample_factor),
        "n_channels": len(artifact_raw.ch_names),
        "n_samples": int(artifact_raw.n_times),
        "sfreq": float(artifact_raw.info["sfreq"]),
        "trigger_count": int(triggers.size),
        "artifact_length_samples": None if artifact_length is None else int(artifact_length),
        "artifact_to_trigger_offset_seconds": float(context.metadata.artifact_to_trigger_offset),
        "history": [step.name for step in context.get_history()],
        "note": (
            "Combined AAS + OBS/PCA artifact (run_3 Weg A, Hebel 1). Full bandwidth, "
            "no 70 Hz low-pass, no down-sample. OBS hp_freq keeps the EEG band out of "
            "the basis (no brain removal); the OBS contribution is the >hp_freq residual."
        ),
    }
    (args.output_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved combined AAS + PCA(OBS) artifact bundle:")
    print(f"  npz       : {npz_path}")
    print(f"  metadata  : {args.output_dir / METADATA_NAME}")
    print(f"  artifact  : AAS + PCACorrection(n_components={args.n_components}, hp_freq={args.hp_freq})")
    print(f"  channels  : {len(artifact_raw.ch_names)}")
    print(f"  samples   : {artifact_raw.n_times}")
    print(f"  sfreq     : {artifact_raw.info['sfreq']} Hz")
    print(f"  mean |art|: {float(np.mean(np.abs(artifact_raw.get_data()))) * 1e6:.3f} uV")


if __name__ == "__main__":
    main()
