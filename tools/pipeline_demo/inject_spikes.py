"""Inject known synthetic spikes into the uncorrected recording.

The same waveform is added to every EEG channel. This deliberately uniform
spatial pattern tests correction behavior; it is not a physiological field model.
The thesis uses four nominal 100 microvolt injections and evaluates Fp1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mne
import numpy as np

from facet.misc.eeg_generator import SpikeParams, generate_spike_wave_complex

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=REPO / "examples/datasets/NiazyFMRI.edf")
    ap.add_argument("--out", type=Path, default=REPO / "output/spike_test/NiazyFMRI_spikes.edf")
    ap.add_argument("--at", type=float, nargs="+", default=[30.0, 31.5, 33.0, 34.5], help="At.")
    ap.add_argument("--amplitude", type=float, default=100.0, help="Amplitude.")
    ap.add_argument("--duration-ms", type=float, default=50.0, help="Duration ms.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    raw = mne.io.read_raw_edf(str(args.input), preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    if len(picks) == 0:
        picks = [
            i for i, n in enumerate(raw.ch_names) if n.upper() not in {"EKG", "ECG", "EMG", "EOG", "STATUS", "MARKER"}
        ]
    params = SpikeParams(
        spike_duration_ms=args.duration_ms,
        amplitude_variability=0.0,
        duration_variability=0.0,
        polyspike_probability=0.0,
    )
    complex_uv, centre = generate_spike_wave_complex(sfreq, params, amplitude=args.amplitude)
    complex_v = complex_uv * 1e-06
    placed = []
    for t in args.at:
        start = int(round(t * sfreq)) - centre
        stop = start + complex_v.size
        if start < 0 or stop > raw.n_times:
            print(f"  skipped: {t} s liegt zu nah am Rand")
            continue
        raw._data[np.ix_(picks, np.arange(start, stop))] += complex_v
        placed.append({"t_s": t, "peak_sample": start + centre, "start_s": start / sfreq, "stop_s": stop / sfreq})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(args.out), raw, fmt="edf", overwrite=True, verbose=False)
    truth = {
        "input": os.path.relpath(args.input.resolve(), REPO),
        "sfreq_hz": sfreq,
        "n_channels_injected": len(picks),
        "amplitude_uv": args.amplitude,
        "duration_ms": args.duration_ms,
        "seed": args.seed,
        "complex_samples": int(complex_v.size),
        "centre_offset_samples": int(centre),
        "spikes": placed,
        "note": "Gleiche Amplitude auf allen EEG-Kanälen — bewusst unphysiologisch, siehe Modul-Docstring.",
    }
    truth_path = args.out.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth, indent=1, ensure_ascii=False))
    peak = float(np.abs(complex_uv).max())
    print(
        f"{len(placed)} Spikes gesetzt auf {len(picks)} Kanälen, Spitzenwert {peak:.1f} µV, Länge {complex_v.size / sfreq * 1000:.0f} ms"
    )
    for p in placed:
        print(f"   {p['t_s']:6.2f} s   Sample {p['peak_sample']}")
    print(f"\nWritten: {args.out}\n             {truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
