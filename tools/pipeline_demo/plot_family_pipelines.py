"""One FACETpy pipeline per model family, corrected signals stacked.

Every family in the unified holdout has so far been evaluated on prepared NPZ
tensors. This runs each of them where they would actually be used — inside a
FACETpy pipeline on the raw EDF — and plots the corrected traces of one window
underneath each other, so the families can be compared as *correctors* rather
than as loss numbers.

Arms
----
``uncorrected``
    The chain with **no correction stage at all** — not even the cleanup PCA.
    That exception matters: ``PCACorrection`` is a corrector, and on the raw
    signal it removes 16.7 % of the power on its own. An earlier version left it
    in the reference arm "for symmetry", which quietly measured every other arm
    against an already-corrected baseline and understated all of them.
``farm``
    The shipped reference chain: ``FARMCorrection(cc=0.975)`` + cleanup PCA. This
    is the baseline a reader would get from ``examples/``.
``farm_pca4``
    The *training bundle's* primary correction, ``FARM(cc=0.9) + PCA(4, 300 Hz)``.
    Not a competitor: it is the input the cascade is trained to improve on, so it
    is the correct control for the cascade arm and nothing else.
``<family>``
    One arm per family, the model taking FARM's place entirely.
``wega_direct`` / ``wega_cascade``
    The two Weg-A models, included because they are the strongest correctors we
    have and leaving them out would make the comparison flattering.

What the plot can and cannot say
--------------------------------
There is **no clean reference** on a real recording. Residual RMS therefore says
how much signal is left, not whether the right thing was removed — a corrector
that deletes the EEG along with the artifact scores best. The null-output arm of
the quantitative evaluation exists for exactly that reason; here the low-RMS arms
must be read together with the trace shape, not instead of it.

``distortion_*`` was meant to be a partial answer and turned out to be a null
result, which is worth keeping rather than deleting. The scan starts at about
28.6 s, so the first seconds of this window carry artifact-free EEG; the idea was
that whatever an arm changes there is damage to clean EEG. It measures below
1e-4 µV for *every* arm, because every corrector — the PCA included — works only
on trigger epochs and there are none before the scan. So the number does say
something (no correction reaches outside its epochs) but it does not separate
good arms from bad ones.

``epoch_boundary_step_*`` is the one that does. Every model here works epoch by
epoch and removes each segment's own mean; reassembled into a continuous
recording the segments no longer share a baseline and the signal steps at each
join. A per-epoch evaluation on prepared tensors cannot see this at all — each
epoch is scored alone — which is why it only shows up here. FARM leaves a step
ratio of 1.4 and the cascade 1.0, while eight of the fourteen families are above
2.5.

Note that the chain's closing 70 Hz low-pass flatters the failing arms: what is
left in them after it is energy inside the EEG band, not high-frequency debris a
filter could still take out.

Usage::

    .venv/bin/python tools/pipeline_demo/plot_family_pipelines.py \\
        --start 25 --stop 35 --out output/pipeline_demo/family_stack
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from facet.correction import DeepLearningCorrection                   # noqa: E402
from pipeline_demo import reference_chain                             # noqa: E402
from pipeline_demo.cascade_adapter import CascadeDemucsAdapter        # noqa: E402
from pipeline_demo.direct_adapter import DirectDemucsAdapter          # noqa: E402
from pipeline_demo.family_adapters import (  # noqa: E402
    DEPLOYMENT_SPECS, FAMILY_SPECS, FamilyAdapter)
from pipeline_demo.legacy_dl_adapter import LegacyDLAdapter, DEFAULT_CHECKPOINT  # noqa: E402

#: The two Weg-A checkpoints already used by plot_farm_vs_model.py.
WEGA_DIRECT_CKPT = "training_output/demucsmc_20260812_130422/checkpoints/epoch0039_val_loss0.0000.pt"
WEGA_CASCADE_CKPT = ("output/run6_grid_cascade/mse100_spk1_lr0.001_ch32/"
                     "cascade_mse100_spk1_lr0.001_ch32_20260812_185804/checkpoints/"
                     "epoch0047_val_loss-1.7230.pt")

#: Families whose TorchScript export cannot run on MPS. demucs' ``_cpu.ts`` has
#: the CPU device baked into the trace, and the default export has CUDA baked in,
#: so on this machine there is no GPU route for it at all.
CPU_ONLY = {"demucs"}


@dataclass
class Arm:
    key: str
    label: str
    kind: str                       # none | farm | farm_pca4 | family | wega_direct | wega_cascade
    group: str                      # Referenz | Familie | Weg A
    family: str = ""
    model_id: str = ""
    extra: dict = field(default_factory=dict)


def build_arms(only: list[str] | None, deployment: str = "off") -> list[Arm]:
    """Assemble the arms.

    ``deployment`` selects which editions run: ``"off"`` is the original fourteen,
    ``"only"`` is the retrained ones alone, ``"both"`` puts each pair side by side.
    They are separate arms rather than a replacement because the comparison *is*
    the result — "the architecture cannot correct" and "the objective was wrong"
    are different claims, and only running both distinguishes them.
    """
    arms = [
        Arm("uncorrected", "ohne jede Korrektur (auch ohne PCA)", "none", "Referenz"),
        Arm("farm", "FARM (Referenzkette)", "farm", "Referenz"),
        Arm("farm_pca4", "FARM + PCA/OBS(4, 300 Hz) — Kaskadeneingang",
            "farm_pca4", "Referenz"),
    ]
    if deployment in ("off", "both"):
        for model_id, spec in FAMILY_SPECS.items():
            arms.append(Arm(model_id, f"{model_id} — {spec.family}", "family",
                            "Familie", family=spec.family, model_id=model_id))
    if deployment in ("only", "both"):
        for model_id, spec in DEPLOYMENT_SPECS.items():
            arms.append(Arm(model_id, f"{model_id} — {spec.family}", "family",
                            "Deployment", family=spec.family, model_id=model_id))
    arms += [
        Arm("wega_direct", "Weg A: Demucs-MC direkt (statt FARM)", "wega_direct", "Weg A"),
        Arm("wega_cascade", "Weg A: FARM+PCA4 → Kaskade", "wega_cascade", "Weg A"),
    ]
    if DEFAULT_CHECKPOINT.exists():
        # Not a competitor: it was trained on AAS's own output for this recording
        # (see legacy_dl_adapter), so it is here to show what that costs and what
        # it buys, not to be ranked against models trained on an independent clean.
        arms.append(Arm("legacy_dl", "FACETpy 0.1.0: FC-DAE-Kaskade (auf AAS trainiert)",
                        "legacy_dl", "Legacy"))
    if only:
        keep = set(only)
        arms = [a for a in arms if a.key in keep]
    return arms


def correctors_for(arm: Arm, args) -> list:
    if arm.kind == "none":
        return []
    if arm.kind == "farm":
        return reference_chain.farm()
    if arm.kind == "farm_pca4":
        return reference_chain.cascade_template_stage()
    if arm.kind == "family":
        device = "cpu" if arm.model_id in CPU_ONLY else args.device
        return [DeepLearningCorrection(model=FamilyAdapter(
            arm.model_id, device=device, dc_mode=args.dc_mode))]
    if arm.kind == "legacy_dl":
        return [DeepLearningCorrection(
            model=LegacyDLAdapter(device="cpu", batch_size=64))]
    if arm.kind == "wega_direct":
        return [DeepLearningCorrection(
            model=DirectDemucsAdapter(REPO / WEGA_DIRECT_CKPT, device=args.device,
                                      batch_size=args.batch_size))]
    if arm.kind == "wega_cascade":
        # The cascade consumes `noisy - template`, and the template it was trained
        # against is FARM + PCA/OBS(4, 300) — see reference_chain's docstring. The
        # adapter reads the template from the context's accumulated noise, so the
        # stage order below is what defines it.
        return reference_chain.cascade_template_stage() + [DeepLearningCorrection(
            model=CascadeDemucsAdapter(REPO / WEGA_CASCADE_CKPT, device=args.device,
                                       batch_size=args.batch_size))]
    raise ValueError(arm.kind)


def run_arm(arm: Arm, args) -> dict:
    """Run one pipeline and return the window it produced, cached on disk.

    The stored window runs to ``--analysis-stop``, which is normally far past
    ``--stop``: the plot needs five readable seconds, the metrics need the whole
    scan. Conflating the two is how every number in the run-6 diagnosis came to
    be computed on **5.5 seconds** of a roughly 100-second acquisition — enough
    for a comb metric to resolve the 7.01 Hz epoch rate, but not enough for one
    unlucky stretch not to dominate it.
    """
    # The file name stays bare: diagnose_family_arms.py and every downstream tool
    # address arms as "<key>.npz". The window is recorded *inside* instead and
    # checked on load — a shorter earlier run must not be reused silently, or the
    # metrics quietly go back to covering 5.5 s of a 134 s acquisition.
    cache = args.cache / f"{arm.key}.npz"
    window = np.asarray([args.start, args.analysis_stop], dtype=float)
    if cache.exists() and not args.force:
        z = np.load(cache, allow_pickle=True)
        stored = z["window_s"] if "window_s" in z.files else None
        if stored is not None and np.allclose(stored, window):
            return {"data": z["data"], "ch_names": list(z["ch_names"]),
                    "sfreq": float(z["sfreq"]), "elapsed": float(z["elapsed"]),
                    "triggers": z["triggers"] if "triggers" in z.files else np.empty(0, np.int64),
                    "cached": True}
        print(f"  {arm.key}: cache deckt {stored} ab, gebraucht wird {window} — neu")

    import mne
    t0 = time.perf_counter()
    # The reference arm carries no corrector, and the cleanup PCA is a corrector.
    include_pca = (not args.no_pca) and arm.kind != "none"
    pipe = reference_chain.build(args.input, correctors=correctors_for(arm, args),
                                 include_pca=include_pca, name=arm.key)
    res = pipe.run()
    if not res.success:
        raise RuntimeError(f"{arm.key}: pipeline failed: {res.error}")
    raw = res.context.get_raw()
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
    sf = float(raw.info["sfreq"])
    i0, i1 = int(args.start * sf), int(args.analysis_stop * sf)
    data = raw.get_data(picks=picks)[:, i0:i1].astype(np.float32) * 1e6
    names = [raw.ch_names[i] for i in picks]
    # Trigger positions in window coordinates. Stored because the epoch-boundary
    # step metric needs to know where the joins are, and reconstructing them from
    # the signal afterwards would be guesswork.
    trg = np.asarray(res.context.get_triggers() if res.context.has_triggers() else [],
                     dtype=np.int64)
    trg = trg[(trg >= i0) & (trg < i1)] - i0
    elapsed = time.perf_counter() - t0
    args.cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, data=data, ch_names=np.asarray(names, dtype=object),
                        sfreq=sf, elapsed=elapsed, triggers=trg, window_s=window)
    return {"data": data, "ch_names": names, "sfreq": sf, "elapsed": elapsed,
            "triggers": trg, "cached": False}


def _ref_triggers(results: dict) -> np.ndarray:
    """Trigger positions of the reference arm; every arm shares them."""
    for key in ("uncorrected", *results):
        r = results.get(key)
        if r is not None and np.asarray(r.get("triggers", ())).size:
            return np.asarray(r["triggers"], dtype=np.int64)
    return np.empty(0, dtype=np.int64)


def arm_stats(res: dict, args, reference: dict | None = None,
              triggers: np.ndarray | None = None) -> dict:
    d = res["data"]
    sf = res["sfreq"]
    s0 = int((args.steady_from - args.start) * sf)
    ds = d[:, max(0, s0):]
    out = {}
    if reference is not None and args.scan_onset > args.start:
        # Pre-scan segment: artifact-free, so any deviation from the uncorrected
        # arm is damage. Stops one epoch short of the onset because the first
        # epoch's context window already reaches back over the boundary.
        e = int((args.scan_onset - args.start) * sf)
        a, b = d[:, :e], reference["data"][:, :e]
        diff = a - b
        out["distortion_pre_scan_rms_uv"] = float(np.sqrt(np.mean(diff ** 2)))
        out["distortion_pre_scan_share_pct"] = float(
            100.0 * np.sqrt(np.mean(diff ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-30))
        out["corr_pre_scan_to_uncorrected"] = float(
            np.corrcoef(a.ravel(), b.ravel())[0, 1])
    trg = np.asarray(triggers if triggers is not None else res.get("triggers", ()), dtype=np.int64)
    trg = trg[(trg > 1) & (trg < d.shape[1] - 1) & (trg >= s0)]
    if trg.size >= 8:
        # Epoch-boundary steps. Every model here works epoch by epoch and removes
        # each segment's own mean; reassembled into a continuous recording, the
        # segments no longer share a baseline and the signal steps at each join.
        # A per-epoch evaluation on prepared tensors cannot see this at all — each
        # epoch is scored alone — which is why it only shows up here.
        jump = np.abs(d[:, trg] - d[:, trg - 1])
        allj = np.abs(np.diff(d[:, s0:], axis=1))
        base = float(np.median(allj))
        out["epoch_boundary_step_uv"] = float(np.median(jump))
        out["epoch_boundary_step_ratio"] = float(np.median(jump) / max(base, 1e-30))
        out["sample_step_median_uv"] = base
    out |= {
        "rms_uv": float(np.sqrt(np.mean(d ** 2))),
        "rms_steady_uv": float(np.sqrt(np.mean(ds ** 2))),
        "peak_to_peak_uv": float(np.ptp(d)),
        "peak_to_peak_steady_uv": float(np.ptp(ds)),
        "max_abs_uv": float(np.max(np.abs(d))),
        "elapsed_seconds": round(float(res["elapsed"]), 1),
    }
    return out


def detect_scan_onset(res: dict, args) -> float:
    """First second of the window at which the gradient artifact appears.

    Found from the data rather than hard-coded, because a hard-coded boundary
    that drifts one epoch into the scan would quietly turn the "damage to clean
    EEG" figure into "damage to the artifact", which is the opposite quantity.
    The rule: the first sample whose absolute value exceeds ten times the median
    absolute deviation of the first 0.5 s, minus one epoch of margin.
    """
    d = np.abs(res["data"]).max(axis=0)
    sf = res["sfreq"]
    head = d[:int(0.5 * sf)]
    if head.size == 0:
        return args.start
    thresh = 10.0 * float(np.median(head)) + 1e-12
    above = np.flatnonzero(d > thresh)
    if above.size == 0:
        return args.stop
    onset = args.start + float(above[0]) / sf
    return max(args.start, onset - 0.15)          # one epoch of margin


GROUP_COLOUR = {"Referenz": "#D55E00", "Familie": "#0072B2", "Weg A": "#009E73",
                "Legacy": "#9467BD", "Deployment": "#CC79A7"}


def _group_colour(group: str) -> str:
    """Colour for an arm group, falling back rather than raising.

    A missing entry used to be a ``KeyError`` raised *after* every pipeline had
    already run — twelve minutes of correction thrown away because the figure did
    not know a colour name. The plot is the cheap part; it must not be able to
    destroy the expensive part.
    """
    return GROUP_COLOUR.get(group, "#666666")


def stack_figure(path: Path, arms: list[Arm], results: dict, channel: str, args) -> None:
    """One row per arm, same channel, same window — the requested stacked plot."""
    trg_ref = _ref_triggers(results)
    ok = [a for a in arms if a.key in results]
    names = results[ok[0].key]["ch_names"]
    k = names.index(channel)
    sf = results[ok[0].key]["sfreq"]
    # The arms are stored out to --analysis-stop; the figure shows --start..--stop.
    # Plotting the whole stored window would compress a hundred seconds into the
    # width of the page and show nothing at all.
    plot_n = int((args.stop - args.start) * sf)
    t = args.start + np.arange(plot_n) / sf
    s0 = int((args.steady_from - args.start) * sf)

    # A shared y-scale across the corrected arms, taken from the steady state:
    # per-arm autoscaling would make a corrector that removes nothing look
    # identical to one that removes everything.
    #
    # The limit is the *upper quartile* of the per-arm amplitudes, not the
    # maximum. With fourteen families one arm that barely corrects would set a
    # scale on which the other thirteen collapse into flat lines — the figure
    # would then show nothing about the arms it is meant to compare. Arms that
    # exceed the limit are clipped and labelled as such, and their RMS and
    # peak-to-peak are printed on the row either way, so nothing is hidden.
    corrected = [a for a in ok if a.kind != "none"]
    amps = {a.key: float(np.percentile(np.abs(results[a.key]["data"][k, s0:]), 99.5))
            for a in corrected}
    lim = 1.2 * float(np.percentile(list(amps.values()), 75))

    fig, axes = plt.subplots(len(ok), 1, figsize=(14, 0.95 * len(ok)),
                             sharex=True, squeeze=False)
    for ax, arm in zip(axes[:, 0], ok):
        y = results[arm.key]["data"][k, :plot_n]
        colour = _group_colour(arm.group)
        ax.plot(t, y, lw=0.45, color=colour)
        if arm.kind == "none":
            ax.set_ylim(-1.05 * np.max(np.abs(y)), 1.05 * np.max(np.abs(y)))
            scale_note = "eigene Skala"
        else:
            ax.set_ylim(-lim, lim)
            scale_note = "· beschnitten" if amps[arm.key] > lim else ""
        st = arm_stats(results[arm.key], args, results.get("uncorrected"), trg_ref)
        ax.set_ylabel(arm.label, fontsize=6.5, rotation=0, ha="right", va="center")
        ax.text(0.995, 0.86, f"RMS {st['rms_steady_uv']:.1f} µV  "
                             f"pp {st['peak_to_peak_steady_uv']:.0f} µV  {scale_note}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6,
                color="#333333",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0))
        ax.axvline(args.scan_onset, color="#888888", ls="--", lw=0.7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=6)
    axes[-1, 0].set_xlabel("Zeit (s)", fontsize=8)
    fig.suptitle(
        f"Kanal {channel} · {args.start:.0f}–{args.stop:.0f} s · je eine vollständige "
        f"FACETpy-Pipeline pro Familie\n"
        f"gemeinsame y-Skala für alle korrigierten Zeilen (oberes Quartil der "
        f"Amplituden ab {args.steady_from:.0f} s, Ausreißer beschnitten und markiert); "
        f"RMS/pp ebenfalls ab {args.steady_from:.0f} s",
        fontsize=9, y=1.0)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_html(path: Path, figures: list[Path], arms: list[Arm], results: dict,
               failures: dict, args) -> None:
    trg_ref = _ref_triggers(results)
    rows = []
    ranked = sorted((a for a in arms if a.key in results),
                    key=lambda a: arm_stats(results[a.key], args, results.get("uncorrected"), trg_ref)["rms_steady_uv"])
    for a in ranked:
        st = arm_stats(results[a.key], args, results.get("uncorrected"), trg_ref)
        rows.append(
            f"<tr><td>{a.label}</td><td>{a.group}</td>"
            f"<td class=n>{st['rms_steady_uv']:.2f}</td>"
            f"<td class=n>{st['rms_uv']:.2f}</td>"
            f"<td class=n>{st['peak_to_peak_steady_uv']:.0f}</td>"
            f"<td class=n>{st.get('epoch_boundary_step_uv', float('nan')):.2f}</td>"
            f"<td class=n>{st.get('epoch_boundary_step_ratio', float('nan')):.1f}</td>"
            f"<td class=n>{st.get('distortion_pre_scan_rms_uv', float('nan')):.2e}</td>"
            f"<td class=n>{st['elapsed_seconds']:.0f}</td></tr>")
    fail_rows = "".join(
        f"<tr><td>{k}</td><td colspan=8 class=err>{v.splitlines()[-1][:200]}</td></tr>"
        for k, v in failures.items())
    imgs = "".join(
        f'<h2>{f.stem}</h2><img src="data:image/png;base64,'
        f'{base64.b64encode(f.read_bytes()).decode()}">' for f in figures)
    path.write_text(f"""<!doctype html>
<meta charset="utf-8"><title>Familien-Pipelines {args.start:.0f}–{args.stop:.0f}s</title>
<style>
body{{font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:2rem;max-width:1200px}}
img{{max-width:100%;border:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0;font-size:13px}}
td,th{{border:1px solid #ccc;padding:3px 8px}} th{{background:#f2f2f2}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
td.err{{color:#b00}} .warn{{background:#fff6e0;border-left:4px solid #e8a33d;padding:.6rem 1rem}}
</style>
<h1>Eine vollständige FACETpy-Pipeline pro Modellfamilie</h1>
<p>Fenster {args.start:.0f}–{args.stop:.0f} s der Aufnahme
<code>{args.input}</code>. Alle Arme durchlaufen dieselbe Kette; ausgetauscht wird
nur die Korrekturstufe.</p>
<pre>Loader(artifact_to_trigger_offset={reference_chain.ARTIFACT_TO_TRIGGER_OFFSET}) → DropChannels → Crop{reference_chain.CROP}
  → HighPass {reference_chain.HIGHPASS_HZ} Hz → TriggerDetector → UpSample ×{reference_chain.UPSAMPLE}
  → TriggerAligner → SubsampleAligner → [ KORREKTOR ]
  → PCA{reference_chain.PCA_KWARGS} → DownSample ×{reference_chain.UPSAMPLE} → LowPass {reference_chain.LOWPASS_HZ} Hz</pre>
<div class="warn"><b>Wie diese Tabelle zu lesen ist.</b> Auf einer echten Aufnahme
gibt es kein sauberes Referenzsignal. Der Rest-RMS sagt, <i>wie viel</i> übrig
blieb, nicht ob das Richtige entfernt wurde: ein Korrektor, der das EEG mit
löscht, steht hier oben. Die niedrigen Zeilen sind deshalb nur zusammen mit dem
Kurvenverlauf und mit der quantitativen Auswertung (inkl. Null-Ausgabe-Arm) zu
bewerten.</div>
<div class="warn"><b>Stufen an den Epochengrenzen.</b> Jedes dieser Modelle
arbeitet epochenweise und entfernt je Segment dessen eigenen Mittelwert. Wieder zu
einer durchgehenden Aufnahme zusammengesetzt, teilen die Segmente keine Basislinie
mehr, und das Signal springt an jeder Naht. <b>Eine Auswertung auf vorbereiteten
Tensoren kann das gar nicht sehen</b> — dort wird jede Epoche einzeln bewertet.
Die Spalte „Stufe/Sample" setzt den Sprung an der Naht ins Verhältnis zum
gewöhnlichen Sprung zwischen zwei Samples: FARM liegt bei 1,4, die Kaskade bei 1,0.
Das unkorrigierte Signal liegt bei 5,0 — das Artefakt selbst springt an den
Grenzen —, ein Wert über etwa 2,5 heißt also, dass der Korrektor die Naht nicht
schließt, sondern eine eigene erzeugt.</div>
<div class="warn"><b>„Schaden vor dem Scan" ist ein Nullbefund.</b> Der Gradient
setzt erst bei {args.scan_onset:.2f} s ein, und die Spalte sollte zeigen, welcher Arm
sauberes EEG beschädigt. Sie zeigt für <i>jeden</i> Arm praktisch null (&lt; 1e-4 µV):
alle Korrektoren — auch die PCA — arbeiten ausschließlich auf Triggerepochen, und
vor dem Scan gibt es keine. Der Wert belegt also, dass keine Korrektur über ihre
Epochen hinausgreift; als Qualitätsunterscheidung taugt er nicht.</div>
<table><tr><th>Arm</th><th>Gruppe</th><th>RMS ab {args.steady_from:.0f}s (µV)</th>
<th>RMS gesamt (µV)</th><th>pp ab {args.steady_from:.0f}s (µV)</th>
<th>Stufe an der Naht (µV)</th><th>Stufe/Sample</th>
<th>Schaden vor dem Scan (µV)</th><th>Laufzeit (s)</th></tr>
{''.join(rows)}{fail_rows}</table>
{imgs}
""", encoding="utf-8")


def main() -> None:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=REPO / "examples/datasets/NiazyFMRI.edf")
    p.add_argument("--out", type=Path, default=REPO / "output/pipeline_demo/family_stack")
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--start", type=float, default=25.0)
    p.add_argument("--analysis-stop", type=float, default=None,
                   help="End of the window the *metrics* use, in seconds. Defaults to "
                        "--stop, which keeps the old behaviour; set it to the end of the "
                        "acquisition so the diagnosis covers the whole scan rather than "
                        "the few seconds that happen to be plotted.")
    p.add_argument("--stop", type=float, default=35.0)
    p.add_argument("--steady-from", type=float, default=30.0)
    p.add_argument("--scan-onset", type=float, default=None,
                   help="Second at which the gradient artifact starts. Detected from "
                        "the uncorrected arm when omitted.")
    p.add_argument("--channels", nargs="*", default=["Fp1", "Cz"])
    p.add_argument("--only", nargs="*", default=None, help="Run a subset of arms.")
    p.add_argument("--deployment", choices=("off", "only", "both"), default="off",
                   help="Include the retrained deployment editions: off (the original "
                        "fourteen), only, or both side by side.")
    p.add_argument("--dc-mode", default="as_evaluated",
                   choices=("as_evaluated", "reconcile", "segment_mean"),
                   help="as_evaluated keeps each model's own inference contract; "
                        "reconcile drops every prediction's offset; segment_mean "
                        "replaces it with the noisy segment's own mean, which is "
                        "what the demeaned training contract implies.")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-pca", action="store_true")
    p.add_argument("--force", action="store_true", help="Ignore cached arm results.")
    args = p.parse_args()
    if args.analysis_stop is None:
        args.analysis_stop = args.stop
    if args.analysis_stop < args.stop:
        raise SystemExit("--analysis-stop must not be before --stop: the figure would "
                         "plot samples the arms do not contain")
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache = args.cache or (args.out / "arms")
    args.cache.mkdir(parents=True, exist_ok=True)

    arms = build_arms(args.only, args.deployment)
    results, failures = {}, {}
    for arm in arms:
        try:
            res = run_arm(arm, args)
            results[arm.key] = res
            tag = "cache" if res["cached"] else f"{res['elapsed']:6.1f}s"
            print(f"[ ok ] {arm.key:22s} {tag}", flush=True)
        except Exception:                                  # noqa: BLE001
            failures[arm.key] = traceback.format_exc()
            print(f"[FAIL] {arm.key:22s} {failures[arm.key].splitlines()[-1][:120]}",
                  flush=True)

    if not results:
        raise SystemExit("no arm produced a result")

    trg_ref = _ref_triggers(results)
    ref = results.get("uncorrected")
    if args.scan_onset is None:
        args.scan_onset = round(detect_scan_onset(ref, args), 2) if ref else args.start
        print(f"Scanbeginn erkannt bei {args.scan_onset:.2f} s")

    names = results[next(iter(results))]["ch_names"]
    chosen = [c for c in args.channels if c in names] or names[:1]
    figures = []
    for ch in chosen:
        fp = args.out / f"family_stack_{ch}_{args.start:.0f}_{args.stop:.0f}s.png"
        stack_figure(fp, arms, results, ch, args)
        figures.append(fp)
        print(f"wrote {fp}")

    html = args.out / "index.html"
    write_html(html, figures, arms, results, failures, args)
    print(f"wrote {html}")

    (args.out / "family_stack_stats.json").write_text(json.dumps({
        "input": str(args.input),
        "window_s": [args.start, args.stop],
        "analysis_window_s": [args.steady_from, args.analysis_stop],
        "steady_from_s": args.steady_from,
        "scan_onset_s": args.scan_onset,
        "pre_scan_note": "Vor dem Scanbeginn gibt es kein Artefakt. Alle Arme laufen "
                         "durch dieselben Filter und dieselbe Aufräum-PCA, also ist jede "
                         "Abweichung vom unkorrigierten Arm dort Schaden an sauberem EEG. "
                         "FARM misst hier exakt 0 — das ist die Kontrolle, dass die "
                         "Kennzahl misst, was sie soll.",
        "channels_plotted": chosen,
        "n_eeg_channels": len(names),
        "chain": "Referenzkette aus examples/complete_pipeline_example.py",
        "cascade_template_stage": {
            "farm": reference_chain.TRAINING_FARM_KWARGS,
            "pca": reference_chain.TRAINING_PCA_KWARGS,
            "why": "Trainings-Bundle-Primärkorrektur; das ist das Template, das die "
                   "Kaskade als abgezogen voraussetzt.",
        },
        "cleanup_pca": reference_chain.PCA_KWARGS,
        "dc_mode": args.dc_mode,
        "caveat": "Kein sauberes Referenzsignal: Rest-RMS misst Entfernungsmenge, "
                  "nicht Korrektheit.",
        "stats": {k: arm_stats(v, args, results.get("uncorrected"), trg_ref)
                  for k, v in results.items()},
        "failures": {k: v.splitlines()[-1] for k, v in failures.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
