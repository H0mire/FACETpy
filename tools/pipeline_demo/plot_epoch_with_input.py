"""Eine Artefaktepoche: die Korrektur, und darunterliegend der rohe Eingang.

``compare_against_legacy.py`` zeigt die Arme auf **gemeinsamer** y-Skala, damit
sie untereinander vergleichbar sind. Das unkorrigierte Signal passt dort nicht
hinein: es führt rund +-400 µV, die Korrekturen +-40. Auf einer Achse wäre die
Korrektur eine waagerechte Linie.

Hier bekommt jeder Arm deshalb **zwei** Achsen: links die Korrektur in ihrer
eigenen Skala, rechts der rohe Eingang in seiner. Die Skalen sind über alle
Tafeln hinweg gleich, damit die Tafeln untereinander weiter vergleichbar
bleiben -- nur nicht die linke gegen die rechte.

Das Fenster ist triggerausgerichtet: es beginnt an einem Trigger und ist genau
eine Epoche lang, also liegt an beiden Rändern eine Naht.

Nutzung::

    uv run python tools/pipeline_demo/plot_epoch_with_input.py \\
        --arms farm nested_gan_tuned demucs_tuned vit_spectrogram_tuned \\
        --trigger-index 40 --channel Fp1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

#: Ab hier ist die Aufnahme eingeschwungen (Armzeit, der Arm beginnt bei 25 s).
EINGESCHWUNGEN_AB_S = 4.5

LABELS = {"farm": "FARM / AAS (Referenz)", "uncorrected": "unkorrigiert"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-dir", type=Path,
                    default=REPO / "output/pipeline_demo/run8_tuned/arms")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--input-arm", default="uncorrected",
                    help="der Arm, der als roher Eingang hinterlegt wird")
    ap.add_argument("--channel", default="Fp1")
    ap.add_argument("--trigger-index", type=int, default=40,
                    help="der wievielte Trigger im eingeschwungenen Bereich")
    ap.add_argument("--epochs", type=float, default=1.0, help="wie viele Epochen breit")
    ap.add_argument("--out", type=Path,
                    default=REPO / "output/pipeline_demo/run8_tuned/epoche_mit_eingang.png")
    args = ap.parse_args()

    def lade(name: str):
        b = np.load(args.arm_dir / f"{name}.npz", allow_pickle=True)
        n = [str(x) for x in b["ch_names"]]
        if args.channel not in n:
            raise SystemExit(f"{name}: Kanal {args.channel!r} fehlt")
        return (b["data"][n.index(args.channel)].astype(np.float64),
                float(b["sfreq"]), b["triggers"])

    roh, sf, trig = lade(args.input_arm)
    tt = trig[trig >= int(EINGESCHWUNGEN_AB_S * sf)]
    if args.trigger_index >= len(tt) - 2:
        raise SystemExit(f"nur {len(tt)} Trigger im eingeschwungenen Bereich")
    i0 = int(tt[args.trigger_index])
    laenge = float(np.diff(tt[:60]).mean())          # Epochenlänge in Samples
    i1 = i0 + int(round(laenge * args.epochs))
    t = np.arange(i0, i1) / sf + 25.0                # Aufnahmezeit
    nahtstellen = [x / sf + 25.0 for x in tt if i0 <= x <= i1]

    spuren = [(a, lade(a)[0][i0:i1]) for a in args.arms]
    korr_max = max(float(np.abs(s).max()) for _, s in spuren) * 1.15
    roh_max = float(np.abs(roh[i0:i1]).max()) * 1.15

    fig, achsen = plt.subplots(len(spuren), 1, figsize=(12, 2.1 * len(spuren)),
                               sharex=True)
    achsen = np.atleast_1d(achsen)
    for ax, (name, sig) in zip(achsen, spuren, strict=True):
        ax2 = ax.twinx()
        ax2.plot(t, roh[i0:i1], color="0.78", lw=0.9, zorder=1)
        ax2.set_ylim(-roh_max, roh_max)
        ax2.set_ylabel("Eingang µV", color="0.55", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="0.55", labelsize=7)
        farbe = "#3b3b8f" if name == "farm" else "#d4541e"
        ax.plot(t, sig, color=farbe, lw=1.2, zorder=3)
        ax.set_ylim(-korr_max, korr_max)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_ylabel(LABELS.get(name, name), fontsize=8)
        for x in nahtstellen:
            ax.axvline(x, color="0.4", lw=0.7, ls=(0, (4, 3)), zorder=2)
        ax.grid(alpha=0.25, lw=0.5)
    achsen[-1].set_xlabel("Zeit (s)")
    dauer_ms = (i1 - i0) / sf * 1000
    fig.suptitle(f"Kanal {args.channel} · {dauer_ms:.0f} ms = {args.epochs:g} Artefaktepoche"
                 f" · grau: unkorrigierter Eingang (rechte Achse)\n"
                 f"gestrichelt: Trigger / Epochennaht", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115)
    print(f"  Eingang  +-{roh_max/1.15:7.1f} µV   Korrekturen  +-{korr_max/1.15:6.1f} µV"
          f"   Verhaeltnis {roh_max/korr_max:.0f}:1")
    print(f"geschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
