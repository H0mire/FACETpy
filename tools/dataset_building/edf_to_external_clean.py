"""Turn a corrected EDF into the ``external_clean`` array the Weg-A builder wants.

Why. The builder's ``niazy_pretrigger`` clean source is the GA-free segment
*before* the first trigger — in-scanner EEG that still carries the
ballistocardiogram, and only ~27 s long. Both properties limit what a
spike-preservation study can say:

* The BCG sits in the *clean target*, so every metric treats a ~67 µV
  peak-to-peak pulse artifact as signal to preserve — larger than the injected
  IEDs themselves.
* 27 s of clean forces an implausibly high IED rate to reach enough independent
  spike events for a paired test.

A recording with GA *and* BCG removed fixes both: no pulse artifact in the
target, and the full ~162 s available. This tool converts such a recording into
the exact array shape ``clean_source='external'`` requires.

What it reproduces from the pre-trigger path, so the two clean sources stay
comparable: non-EEG channels dropped, 1 Hz high-pass, mains notch with harmonics,
band-limited polyphase resampling to the bundle rate.

What it adds: the channel set and **order** are taken from the artifact bundle,
not from the EDF. Silently feeding a differently ordered montage would pair every
electrode's clean with another electrode's artifact — a defect no metric would
reveal.

Length handling. ``external`` demands an exact match with the artifact array. If
the clean is shorter, the remainder is filled by reflecting material **from the
same side of the train/val boundary**, so no validation background is ever
constructed from training material (the same invariant the builder's pre-trigger
path maintains by splitting before tiling).

Usage::

    .venv/bin/python tools/dataset_building/edf_to_external_clean.py \\
        --input .../NiazyFMRI_ga_and_bcg_removed.edf \\
        --artifact-bundle output/artifact_libraries/niazy_farm_pca4_direct/niazy_aas_pca4_artifact.npz \\
        --output output/clean_sources/niazy_ga_bcg_removed_clean.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="EDF with the gradient and pulse artifact removed")
    p.add_argument(
        "--artifact-bundle",
        type=Path,
        required=True,
        help="Defines the montage, order, rate and length the clean must match",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--highpass-hz", type=float, default=1.0, help="Matches the pre-trigger path; 0 disables")
    p.add_argument("--line-freq", type=float, default=50.0, help="Mains notch with harmonics up to Nyquist; 0 disables")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Must equal the builder's --val-fraction: the fill material is taken from "
        "the correct side of that boundary",
    )
    p.add_argument("--drop", nargs="*", default=["EKG", "EMG", "EOG", "ECG", "Status"])
    return p.parse_args()


def _resample(data: np.ndarray, src: float, dst: float) -> np.ndarray:
    if abs(src - dst) < 1e-9:
        return data
    from scipy.signal import resample_poly

    g = gcd(int(round(dst)), int(round(src)))
    return resample_poly(data, int(round(dst)) // g, int(round(src)) // g, axis=-1)


def _fill_to_length(block: np.ndarray, target: int) -> np.ndarray:
    """Extend ``block`` to ``target`` columns by reflecting its own content.

    Reflection rather than tiling, because a tile seam introduces a step
    discontinuity that looks like a sharp transient to a model trained to detect
    sharp transients. Only the block's own samples are used, so this cannot mix
    material across a split boundary.
    """
    n = block.shape[1]
    if n >= target:
        return block[:, :target]
    out = [block]
    have, flip = n, True
    while have < target:
        piece = block[:, ::-1] if flip else block
        take = min(n, target - have)
        out.append(piece[:, :take])
        have += take
        flip = not flip
    return np.concatenate(out, axis=1)[:, :target]


def main() -> None:
    args = parse_args()
    from mne.filter import notch_filter as mne_notch

    from facet import DropChannels, HighPassFilter, load

    with np.load(args.artifact_bundle, allow_pickle=True) as bundle:
        bundle_names = [str(x) for x in bundle["ch_names"]]
        bundle_sfreq = float(bundle["sfreq"][0])
        n_samples = int(bundle["artifact"].shape[1])
    print(
        f"bundle: {len(bundle_names)} channels @ {bundle_sfreq:.0f} Hz, {n_samples} samples "
        f"({n_samples / bundle_sfreq:.1f} s)"
    )

    ctx = load(str(args.input), preload=True)
    ctx = ctx | DropChannels(channels=list(args.drop))
    if args.highpass_hz and args.highpass_hz > 0:
        ctx = ctx | HighPassFilter(freq=args.highpass_hz)
    raw = ctx.get_raw()
    sfreq = float(raw.info["sfreq"])

    missing = [n for n in bundle_names if n not in raw.ch_names]
    if missing:
        raise SystemExit(
            f"the clean recording lacks {len(missing)} of the bundle's channels: {missing}. "
            "Both must use the same montage."
        )
    # Order by the bundle, not by the EDF: this is the invariant that keeps every
    # electrode's clean paired with its own artifact.
    picks = [raw.ch_names.index(n) for n in bundle_names]
    clean = raw._data[picks].astype(np.float64)
    print(
        f"input : {clean.shape[0]} channels @ {sfreq:.0f} Hz, {clean.shape[1]} samples ({clean.shape[1] / sfreq:.1f} s)"
    )

    if args.line_freq and args.line_freq > 0:
        freqs = np.arange(args.line_freq, sfreq / 2.0, args.line_freq)
        if freqs.size:
            clean = mne_notch(clean, sfreq, freqs, verbose="ERROR")
            print(f"  notch @ {', '.join(f'{f:.0f}' for f in freqs)} Hz")

    clean = _resample(clean, sfreq, bundle_sfreq)
    print(f"  resampled to {bundle_sfreq:.0f} Hz -> {clean.shape[1]} samples")

    # Fill each side of the train/val boundary from its own material only.
    split = int(round(n_samples * (1.0 - args.val_fraction)))
    src_split = int(round(clean.shape[1] * (1.0 - args.val_fraction)))
    train = _fill_to_length(clean[:, :src_split], split)
    val = _fill_to_length(clean[:, src_split:], n_samples - split)
    out = np.concatenate([train, val], axis=1).astype(np.float32)
    assert out.shape == (len(bundle_names), n_samples), out.shape

    reflect_train = max(0, split - src_split)
    reflect_val = max(0, (n_samples - split) - (clean.shape[1] - src_split))
    print(
        f"  train side: {src_split} source -> {split} samples ({100 * reflect_train / max(split, 1):.1f} % reflected)"
    )
    print(
        f"  val   side: {clean.shape[1] - src_split} source -> {n_samples - split} samples "
        f"({100 * reflect_val / max(n_samples - split, 1):.1f} % reflected)"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, clean=out, ch_names=np.array(bundle_names, dtype=object), sfreq=np.array([bundle_sfreq])
    )
    digest = hashlib.sha256(np.ascontiguousarray(out)).hexdigest()
    meta = {
        "source_edf": str(args.input),
        "artifact_bundle": str(args.artifact_bundle),
        "channels": bundle_names,
        "channel_order_source": "artifact bundle (not the EDF)",
        "sfreq_hz": bundle_sfreq,
        "n_samples": n_samples,
        "highpass_hz": args.highpass_hz,
        "line_freq_hz": args.line_freq,
        "val_fraction": args.val_fraction,
        "reflected_samples_train": reflect_train,
        "reflected_samples_val": reflect_val,
        "fill_rule": "reflection within the same side of the train/val boundary",
        "clean_sha256": digest,
        "rms_uv": float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) * 1e6,
        "mean_abs_uv": float(np.mean(np.abs(out.astype(np.float64)))) * 1e6,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"  RMS {meta['rms_uv']:.2f} µV, mean|x| {meta['mean_abs_uv']:.2f} µV, sha256 {digest[:16]}")


if __name__ == "__main__":
    main()
