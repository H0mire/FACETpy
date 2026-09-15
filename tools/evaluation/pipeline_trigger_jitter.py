"""Trigger jitter in the deployed pipeline: what imprecise detection costs FARM.

The controlled sweep in ``template_jitter_sweep.py`` isolates the mechanism with
ground truth, but its template is a mean over the six neighbour epochs the
dataset happens to store. Real FARM averages a ~30-epoch window with a
correlation threshold, and a wider average is more forgiving of a single
misaligned epoch. This tool therefore repeats the question where FARM actually
estimates its template — on the EDF, in the pipeline.

Two arms, the same distinction as in the controlled sweep:

* ``estimation`` — each trigger is perturbed independently, so FARM averages
  misaligned epochs and its template comes out smeared. The realistic failure.
* ``application`` — every trigger is perturbed by the *same* offset, so the
  template is estimated correctly from mutually consistent epochs but the whole
  correction sits off. Note this is a *global* offset, which is exactly the case
  the models were trained to tolerate.

There is no clean reference on a real recording, so nothing here is an accuracy
claim. What is measurable is how much artifact power the correction removes and
how much of the residual sits above the EEG band — both descriptive.

.. warning::

   The first version of this tool built its own step list and inherited the
   ``Loader`` default ``artifact_to_trigger_offset = 0.0``. The artifact starts
   5 ms *before* the trigger, so that put part of it outside every epoch window
   and left FARM with roughly three times the residual it should have. Every
   number produced before that fix is void. The chain now comes from
   :mod:`tools.pipeline_demo.reference_chain`, which is the only copy.

Usage::

    .venv/bin/python tools/evaluation/pipeline_trigger_jitter.py \\
        --input examples/datasets/NiazyFMRI.edf --out output/model_evaluations/pipeline_jitter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facet.core import Pipeline, Processor, ProcessingContext   # noqa: E402
from facet.correction import FARMCorrection                     # noqa: E402
from pipeline_demo import reference_chain                       # noqa: E402

#: Jitter magnitudes in **milliseconds**, not samples.
#:
#: The processor sits after ``UpSample``, where one sample is a tenth of a native
#: sample. Sweeping in samples there would have measured a jitter ten times
#: smaller than intended — it did, in the first version of this tool — so the
#: sweep is defined in physical time and converted at run time.
SWEEP_MS = [0.0, 0.06, 0.12, 0.24, 0.49, 0.98, 1.95, 3.91]


class TriggerJitter(Processor):
    """Perturb the detected trigger positions before the template is estimated.

    ``mode="estimation"`` draws one offset per trigger, so the epochs FARM
    averages no longer line up. ``mode="application"`` draws a single offset and
    applies it to every trigger, which keeps the epochs mutually consistent and
    only moves the whole correction.

    The offsets are in samples of the *current* (up-sampled) rate, so the caller
    must place this processor after ``UpSample`` for the numbers to mean what the
    controlled sweep's numbers mean.
    """

    name = "trigger_jitter"
    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, sd_ms: float, mode: str = "estimation", seed: int = 0) -> None:
        self.sd_ms = float(sd_ms)
        self.mode = mode
        self.seed = int(seed)
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        triggers = np.asarray(context.metadata.triggers, dtype=np.int64)
        if self.sd_ms <= 0 or triggers.size == 0:
            return context
        # Convert to samples of the CURRENT rate, which is the up-sampled one.
        sd = self.sd_ms * 1e-3 * context.get_sfreq()
        rng = np.random.default_rng(self.seed)
        if self.mode == "estimation":
            off = np.rint(rng.normal(0.0, sd, size=triggers.size)).astype(np.int64)
        else:
            # A single draw of at least one sample, so the arm actually moves.
            draw = rng.normal(0.0, sd)
            step = int(np.sign(draw) * max(1.0, abs(round(draw)))) if sd > 0 else 0
            off = np.full(triggers.size, step, dtype=np.int64)
        n = context.get_raw().n_times
        jittered = np.clip(triggers + off, 0, n - 1).astype(np.int64)
        # with_triggers, not a metadata edit: the context is immutable by
        # contract, and downstream AAS does arithmetic on the array — handing it
        # a Python list makes FARM fail with a list-concatenation TypeError.
        return context.with_triggers(jittered)


def metrics(raw_before, raw_after) -> dict:
    import mne
    picks = mne.pick_types(raw_after.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
    a = raw_before.get_data(picks=picks) * 1e6
    b = raw_after.get_data(picks=picks) * 1e6
    removed = a - b
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    from scipy.signal import welch
    sf = raw_after.info["sfreq"]
    f, p = welch(b, fs=sf, nperseg=min(4096, b.shape[-1]))
    total = float(p.sum())
    return {
        "rms_raw_uv": rms(a), "rms_corrected_uv": rms(b), "rms_removed_uv": rms(removed),
        "power_removed_pct": 100.0 * (1.0 - rms(b) ** 2 / rms(a) ** 2),
        "above_70hz_power_share_pct": 100.0 * float(p[:, f > 70].sum()) / max(total, 1e-30),
        "eeg_band_power_share_pct": 100.0 * float(p[:, (f >= 1) & (f <= 45)].sum()) / max(total, 1e-30),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("examples/datasets/NiazyFMRI.edf"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trigger-regex", default=r"\b1\b")
    p.add_argument("--upsample", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # The uncorrected reference does not depend on the jitter, so it is run once.
    # Re-running it inside every sweep point cost 32 extra pipelines and, worse,
    # invited the two chains to drift apart.
    res_u = reference_chain.build(args.input, correctors=[], include_pca=False,
                                  name="uncorrected",
                                  trigger_regex=args.trigger_regex).run()
    if not res_u.success:
        raise RuntimeError(f"uncorrected pipeline failed: {res_u.error}")
    raw_uncorrected = res_u.context.get_raw()

    def run(sd_ms: float, mode: str, realign: bool) -> dict:
        """One sweep point on the reference chain.

        PCA is deliberately left out: the question is what trigger jitter does to
        the *template*, and an OBS stage afterwards partly cleans up whatever
        FARM leaves, which would understate exactly the effect being measured.
        """
        farm_kwargs = dict(reference_chain.FARM_KWARGS)
        farm_kwargs["realign_after_averaging"] = realign
        correctors = []
        if sd_ms > 0:
            correctors.append(TriggerJitter(sd_ms=sd_ms, mode=mode, seed=args.seed))
        correctors.append(FARMCorrection(**farm_kwargs))
        res_c = reference_chain.build(args.input, correctors=correctors,
                                      include_pca=False,
                                      name=f"farm_{mode}_{sd_ms}_{realign}",
                                      trigger_regex=args.trigger_regex).run()
        if not res_c.success:
            raise RuntimeError(f"pipeline failed: {res_c.error}")
        return metrics(raw_uncorrected, res_c.context.get_raw())

    rows = []
    for realign in (True, False):
        for mode in ("estimation", "application"):
            for sd_ms in SWEEP_MS:
                m = run(sd_ms, mode, realign)
                rows.append({"arm": mode, "farm_realign_after_averaging": realign,
                             "jitter_sd_ms": sd_ms,
                             "jitter_sd_native_samples": round(sd_ms * 1e-3 * 4096.0, 3), **m})
                print(f"realign={str(realign):5s} {mode:12s} sd={sd_ms:>5.2f} ms  "
                      f"entfernt {m['power_removed_pct']:>6.2f} %  "
                      f"RMS {m['rms_corrected_uv']:>8.2f} µV  "
                      f">70 Hz {m['above_70hz_power_share_pct']:>6.3f} %", flush=True)

    (args.out / "pipeline_trigger_jitter.json").write_text(json.dumps({
        "input": str(args.input),
        "upsample": reference_chain.UPSAMPLE,
        "chain": "tools/pipeline_demo/reference_chain.py (examples-treu), ohne PCA",
        "artifact_to_trigger_offset_s": reference_chain.ARTIFACT_TO_TRIGGER_OFFSET,
        "farm_kwargs": reference_chain.FARM_KWARGS,
        "pca": "bewusst ausgelassen — eine OBS-Stufe nach FARM raeumt einen Teil des "
               "Jitter-Restes weg und wuerde genau den gemessenen Effekt verkleinern.",
        "farm_window_size": reference_chain.FARM_KWARGS["window_size"],
        "farm_realign_note": "FARM justiert die Trigger nach der Mittelung per "
                             "Kreuzkorrelation nach (search_window_factor 3.0 x "
                             "upsampling_factor). Das absorbiert einen globalen Versatz "
                             "bis etwa +/-3 native Samples von selbst — deshalb laeuft der "
                             "Sweep zusaetzlich mit realign_after_averaging=False.",
        "caveat": "Auf einer echten Aufnahme existiert kein sauberes Referenzsignal. Die "
                  "Kennzahlen sind deskriptiv: sie sagen, WIE VIEL entfernt wurde, nicht ob "
                  "das Richtige entfernt wurde.",
        "arms": {
            "estimation": "Offset je Trigger unabhängig — FARM mittelt fehlausgerichtete "
                          "Epochen, das Template wird verwischt. Der realistische Ausfall.",
            "application": "Ein Offset für alle Trigger — Template korrekt geschätzt, aber "
                           "die ganze Korrektur sitzt versetzt.",
        },
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    import csv
    with (args.out / "pipeline_trigger_jitter.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out / 'pipeline_trigger_jitter.csv'}")


if __name__ == "__main__":
    main()
