"""Turn the visual reading of the family stack into measurements.

Looking at nineteen traces gives an immediate and largely correct diagnosis —
"this one deleted the signal", "this one has a step per epoch", "this one only
distorted the artifact". The point of this script is not to replace that reading
but to put a number on each mechanism, so the chapter can state which arm fails
how instead of describing pictures.

Five mechanisms, five metrics, all relative to the FARM reference arm because
there is no clean signal on a real recording:

``comb_rms_uv``
    Residual gradient artifact **in µV**. The artifact is epoch-periodic, so the
    power sitting on the epoch-repetition frequency and its harmonics is what the
    corrector failed to remove. Reported in µV rather than as a share, because a
    share is misleading in exactly the case that matters: an arm that deleted the
    EEG has almost no power left, so nearly all of the little that remains is
    epoch-periodic and its *share* looks catastrophic while its *amount* is
    small. FARM leaves 4.6 µV; that is the yardstick.
``eeg_band_rel_farm`` / ``lowfreq_rel_farm``
    Power in 1–45 Hz and in 1–8 Hz, divided by FARM's. Well below 1 means the
    arm removed EEG, not just artifact. The low band is separate because several
    arms visibly flatten the baseline while keeping faster activity.
``corr_to_farm``
    Correlation with FARM's output over the same samples. Near zero with a low
    RMS is the signature of deletion; near one means the arm agrees with FARM
    about what the EEG is.
``epoch_boundary_step_ratio``
    Step at the epoch join over the ordinary sample-to-sample step. A per-epoch
    evaluation cannot see this at all.

The verdict column is a rule over those numbers, stated in code rather than in
prose so it can be argued with.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def band(psd: np.ndarray, f: np.ndarray, lo: float, hi: float) -> float:
    return float(psd[:, (f >= lo) & (f < hi)].sum())


def comb_share(psd: np.ndarray, f: np.ndarray, f_epoch: float, fmax: float) -> float:
    """Power on the epoch harmonics as a share of the total.

    A tolerance of one bin either side, because the epoch period is not an exact
    integer number of samples and the comb lines are not perfectly on-grid.
    """
    df = f[1] - f[0]
    mask = np.zeros(f.size, dtype=bool)
    k = 1
    while k * f_epoch < fmax:
        mask |= np.abs(f - k * f_epoch) <= 1.5 * df
        k += 1
    total = float(psd.sum())
    return 100.0 * float(psd[:, mask].sum()) / max(total, 1e-30)


def verdict(m: dict) -> list[str]:
    """Every mechanism that applies, not just the first.

    An earlier version returned one label in priority order and got the two most
    important arms wrong: it called the arm that deleted the EEG *and* left the
    artifact "artifact uncorrected", hiding the deletion. Most of these arms fail
    in more than one way at once, so the honest output is a list.
    """
    out: list[str] = []
    if m["dc_share_pct"] > 50:
        out.append("konstanter Versatz dominiert")
    if m["eeg_band_rel_farm"] > 3:
        out.append("Energie hinzugefuegt statt entfernt")
    elif m["eeg_band_rel_farm"] < 0.5:
        out.append("EEG-Band geloescht")
    elif m["lowfreq_rel_farm"] < 0.5:
        out.append("tiefe Frequenzen geloescht")
    if m["comb_rms_uv"] > 3.0 * m["comb_rms_farm_uv"]:
        out.append("Artefakt weitgehend unkorrigiert")
    elif m["comb_rms_uv"] > 1.5 * m["comb_rms_farm_uv"]:
        out.append("Artefaktreste erkennbar")
    if m["epoch_boundary_step_ratio"] > 2.5:
        out.append("Stufen an den Epochennaehten")
    return out or ["vergleichbar mit FARM"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", type=Path,
                   default=REPO / "output/pipeline_demo/family_stack/arms")
    p.add_argument("--out", type=Path,
                   default=REPO / "output/pipeline_demo/family_stack/arm_diagnosis.json")
    p.add_argument("--from-second", type=float, default=29.5,
                   help="Analyse from here on, so the scan-onset transient is out.")
    p.add_argument("--start", type=float, default=25.0)
    p.add_argument("--fmax", type=float, default=70.0)
    args = p.parse_args()

    ref = np.load(args.arms / "uncorrected.npz", allow_pickle=True)
    sf = float(ref["sfreq"])
    trg = np.asarray(ref["triggers"], dtype=np.int64)
    f_epoch = sf / float(np.median(np.diff(trg)))
    s0 = int((args.from_second - args.start) * sf)

    def load(name):
        z = np.load(args.arms / f"{name}.npz", allow_pickle=True)
        return np.asarray(z["data"], dtype=np.float64)[:, s0:]

    farm = load("farm")
    win = np.hanning(farm.shape[1])
    freqs = np.fft.rfftfreq(farm.shape[1], d=1.0 / sf)

    def psd(x):
        return np.abs(np.fft.rfft(x * win, axis=-1)) ** 2

    psd_farm = psd(farm)
    farm_eeg = band(psd_farm, freqs, 1.0, 45.0)
    farm_low = band(psd_farm, freqs, 1.0, 8.0)
    comb_farm = comb_share(psd_farm, freqs, f_epoch, args.fmax)
    rms_farm = float(np.sqrt(np.mean(farm ** 2)))
    comb_rms_farm = rms_farm * np.sqrt(comb_farm / 100.0)

    def step_ratio(d: np.ndarray) -> float:
        """Epoch-seam step over the ordinary sample step.

        Computed here rather than read from the plot tool's stats file: arms added
        outside that tool (the spike-weight pair, for instance) are absent from it
        and silently came out as NaN.
        """
        t = trg[(trg > s0 + 1) & (trg < s0 + d.shape[1] - 1)] - s0
        if t.size < 8:
            return float("nan")
        base = float(np.median(np.abs(np.diff(d, axis=1))))
        return float(np.median(np.abs(d[:, t] - d[:, t - 1])) / max(base, 1e-30))

    rows = []
    for name in sorted(p.stem for p in args.arms.glob("*.npz")):
        d = load(name)
        pd_ = psd(d)
        rms = float(np.sqrt(np.mean(d ** 2)))
        share = comb_share(pd_, freqs, f_epoch, args.fmax)
        m = {
            "arm": name,
            "rms_uv": round(rms, 2),
            "rms_farm_uv": round(rms_farm, 2),
            "comb_rms_uv": round(rms * np.sqrt(share / 100.0), 3),
            "comb_rms_farm_uv": round(comb_rms_farm, 3),
            "comb_share_pct": round(share, 2),
            "eeg_band_rel_farm": round(band(pd_, freqs, 1.0, 45.0) / max(farm_eeg, 1e-30), 3),
            "lowfreq_rel_farm": round(band(pd_, freqs, 1.0, 8.0) / max(farm_low, 1e-30), 3),
            "corr_to_farm": round(float(np.corrcoef(d.ravel(), farm.ravel())[0, 1]), 3),
            "epoch_boundary_step_ratio": round(step_ratio(d), 2),
            "dc_share_pct": round(100.0 * float(np.mean(d.mean(axis=1) ** 2) / np.mean(d ** 2)), 1),
        }
        m["verdict"] = verdict(m)
        rows.append(m)

    rows.sort(key=lambda r: r["comb_rms_uv"])
    out = {
        "epoch_rate_hz": round(f_epoch, 4),
        "analysis_window_s": [args.from_second, 35.0],
        "reference_arm": "farm",
        "caveat": "Kein sauberes Referenzsignal. Alle Verhaeltnisse sind gegen FARM "
                  "gebildet, nicht gegen Grundwahrheit: sie sagen, worin ein Arm von "
                  "FARM abweicht, nicht wer recht hat.",
        "rows": rows,
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    hdr = (f"{'Arm':22s}{'RMS':>8s}{'GA-Rest':>9s}{'EEG/FARM':>10s}{'tief/FARM':>11s}"
           f"{'r(FARM)':>9s}{'Stufe':>7s}{'DC%':>7s}  Befund")
    print(f"Epochenrate {f_epoch:.4f} Hz, FARM laesst {comb_rms_farm:.2f} µV Artefakt stehen"
          f"\n{hdr}\n{'-' * len(hdr)}")
    for r in rows:
        print(f"{r['arm']:22s}{r['rms_uv']:8.1f}{r['comb_rms_uv']:9.2f}"
              f"{r['eeg_band_rel_farm']:10.3f}{r['lowfreq_rel_farm']:11.3f}"
              f"{r['corr_to_farm']:9.3f}{r['epoch_boundary_step_ratio']:7.1f}"
              f"{r['dc_share_pct']:7.1f}  {', '.join(r['verdict'])}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
