"""Plot a few seconds of each pipeline arm so the result can actually be looked at.

Every metric in this project was written by us, which means every metric can be
wrong in a way no other metric will catch. This script exists because that
happened: the residual-artifact threshold in
:mod:`facet.evaluation.correction_verdict` passed three arms whose traces are
dominated by a regular spike train at the epoch rate, and no number in the
diagnosis table said so. Five seconds of signal did.

Two things it enforces, both learned the hard way:

* **The window must be inside the scan.** The stored arms start at 25 s and the
  gradient artifact starts at ~28.7 s, so the first 3.7 seconds are artifact-free
  and identical in every arm. A plot that includes them looks like six perfect
  correctors.
* **The data is already in µV.** Multiplying by 1e6 "to convert from volts"
  produces RMS values in the tens of millions and a plot that is not obviously
  wrong, only wrong.

Usage::

    python tools/pipeline_demo/look_at_arms.py \
        --arms farm wega_cascade ic_unet demucs \
        --channel Fp1 --start 4.0 --duration 5.0 \
        --out output/pipeline_demo/look_farm_vs_models.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ARM_DIR = REPO / "output/pipeline_demo/family_stack/arms"

#: Seconds into the *stored window* at which the gradient artifact begins. The
#: window itself starts at 25 s of the recording and the scan at ~28.7 s.
SCAN_ONSET_IN_WINDOW = 3.7


def load(arm: str, channel: str, arm_dir: Path) -> tuple[np.ndarray, float]:
    path = arm_dir / f"{arm}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run plot_family_pipelines.py first")
    with np.load(path, allow_pickle=True) as b:
        names = [str(n) for n in b["ch_names"]]
        if channel not in names:
            raise ValueError(f"{arm}: no channel {channel!r}; have {names[:6]}...")
        # Already µV. See the module docstring.
        return b["data"][names.index(channel)], float(b["sfreq"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--channel", default="Fp1")
    ap.add_argument("--start", type=float, default=SCAN_ONSET_IN_WINDOW,
                    help="seconds into the stored window (default: the scan onset)")
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--arm-dir", type=Path, default=ARM_DIR)
    ap.add_argument("--out", type=Path,
                    default=REPO / "output/pipeline_demo/look_at_arms.png")
    args = ap.parse_args()

    if args.start + args.duration <= SCAN_ONSET_IN_WINDOW:
        raise SystemExit(
            f"the whole window is before the scan onset at {SCAN_ONSET_IN_WINDOW} s — "
            f"every arm would look perfect")

    signals = {arm: load(arm, args.channel, args.arm_dir) for arm in args.arms}
    sfreq = next(iter(signals.values()))[1]
    n0 = int(args.start * sfreq)
    n1 = n0 + int(args.duration * sfreq)
    t = np.arange(n1 - n0) / sfreq

    fig, axes = plt.subplots(len(args.arms), 1, figsize=(14, 1.9 * len(args.arms)),
                             sharex=True, squeeze=False)
    for ax, arm in zip(axes[:, 0], args.arms, strict=True):
        seg = signals[arm][0][n0:n1]
        ax.plot(t[:seg.size], seg, linewidth=0.6,
                color="#111111" if arm.startswith("farm") else "#c2410c")
        ax.set_ylabel(f"{arm}\nRMS {float(np.sqrt(np.mean(seg ** 2))):.1f} µV", fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("s")
    fig.suptitle(f"{args.channel}, {args.duration:.0f} s ab {25 + args.start:.1f} s "
                 f"— FARM ist die Referenz (schwarz)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    plt.close(fig)

    print(f"{'arm':22s} {'RMS µV':>9s} {'min':>9s} {'max':>9s}")
    for arm in args.arms:
        seg = signals[arm][0][n0:n1]
        print(f"{arm:22s} {float(np.sqrt(np.mean(seg ** 2))):9.2f} "
              f"{seg.min():9.1f} {seg.max():9.1f}")
    print(f"\ngeschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
