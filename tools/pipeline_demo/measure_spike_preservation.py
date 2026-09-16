"""Measure the response of matched corrected recordings to spike injection.

Subtract the no-injection arm from the injected arm to remove background EEG.
Measure each peak within 150 ms of the recorded injection time. Percentages
relative to the nominal injection and the processed uncorrected arm are separate
columns. Original column names are retained for recorded-table compatibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HALF_WINDOW_MS = 150.0


def load(arm_dir: Path, arm: str, channel: str) -> tuple[np.ndarray, float, float]:
    path = arm_dir / f"{arm}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as b:
        names = [str(n) for n in b["ch_names"]]
        if channel not in names:
            raise ValueError(f"{arm}: Channel {channel!r} is missing; available {names[:6]}…")
        return (b["data"][names.index(channel)].astype(np.float64), float(b["sfreq"]), float(b["window_s"][0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-dir", type=Path, default=REPO / "output/spike_test/arms")
    ap.add_argument("--without-dir", type=Path, default=REPO / "output/pipeline_demo/deployment_first/arms")
    ap.add_argument("--truth", type=Path, default=REPO / "output/spike_test/NiazyFMRI_spikes.truth.json")
    ap.add_argument("--channel", default="Fp1")
    ap.add_argument("--out", type=Path, default=REPO / "output/spike_test/spike_preservation.csv")
    args = ap.parse_args()
    truth = json.loads(args.truth.read_text())
    times = [s["t_s"] for s in truth["spikes"]]
    injected = float(truth["amplitude_uv"])
    arms = sorted(p.stem for p in args.with_dir.glob("*.npz"))
    rows = []
    for arm in arms:
        try:
            with_sig, sfreq, offset = load(args.with_dir, arm, args.channel)
            without_sig, sfreq2, offset2 = load(args.without_dir, arm, args.channel)
        except (FileNotFoundError, ValueError) as e:
            print(f"  {arm}: skipped ({e.__class__.__name__})")
            continue
        if sfreq != sfreq2 or abs(offset - offset2) > 1e-06 or with_sig.size != without_sig.size:
            print(f"  {arm}: skipped — the arms do not cover the same window")
            continue
        half = int(HALF_WINDOW_MS * sfreq / 1000)
        peaks = []
        for t in times:
            i = int(round((t - offset) * sfreq))
            lo, hi = (max(0, i - half), min(with_sig.size, i + half))
            peaks.append(float(np.abs(with_sig[lo:hi] - without_sig[lo:hi]).max()))
        rows.append(
            {
                "arm": arm,
                **{f"spike_{t:g}s_uv": round(p, 2) for t, p in zip(times, peaks, strict=True)},
                "mittel_uv": round(float(np.mean(peaks)), 2),
                "erhalt_pct": round(100.0 * float(np.mean(peaks)) / injected, 1),
            }
        )
    referenz = next((r["mittel_uv"] for r in rows if r["arm"] == "uncorrected"), None)
    for r in rows:
        r["erhalt_vs_uncorrected_pct"] = round(100.0 * r["mittel_uv"] / referenz, 1) if referenz else None
    if not rows:
        raise SystemExit("No comparable arms found")
    rows.sort(key=lambda r: -r["erhalt_pct"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(
        f"Injected: {injected:.0f} µV; measured at {args.channel}, {len(times)} locations ({', '.join(f'{t:g} s' for t in times)})\n"
    )
    print(f"{'Arm':32s}" + "".join(f"{f'{t:g}s':>9s}" for t in times) + f"{'Mean':>9s}{'/100µV':>9s}{'/raw':>10s}")
    for r in rows:
        vals = "".join(f"{r[k]:9.1f}" for k in r if k.startswith("spike_"))
        gegen = r.get("erhalt_vs_uncorrected_pct")
        print(
            f"{r['arm']:32s}{vals}{r['mittel_uv']:9.1f}{r['erhalt_pct']:8.1f}%{(f'{gegen:9.1f}%' if gegen is not None else '        -')}"
        )
    print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
