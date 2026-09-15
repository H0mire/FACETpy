"""Fahre die Gewinner der Gittersuche durch dieselbe Kette und speichere den Arm.

Die Gittersuche hat jeden Punkt in der Pipeline bewertet, aber nur die Kennzahl
behalten, nicht das Signal. Für einen Blick auf die Korrektur -- und dieses
Projekt hat zweimal erlebt, dass eine gute Kennzahl ein kaputtes Signal verdeckt
-- braucht es das Signal selbst.

Die Arme landen im selben Format wie die von ``plot_family_pipelines.py``, damit
``compare_against_legacy.py`` und ``measure_arms.py`` sie ohne Sonderfall lesen.

Nutzung::

    uv run python tools/pipeline_demo/run_tuned_arms.py \\
        --arm nested_gan=pfad/zu/export.ts --out output/pipeline_demo/run8_tuned/arms
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "pipeline_demo"))
sys.path.insert(0, str(REPO / "tools" / "training"))

#: Dasselbe Fenster wie run 7 und die Gittersuche. Ein abweichendes Fenster ist
#: ein anderer Maßstab -- der Kammwert haengt von der Fensterlaenge ab.
START_S, STOP_S = 25.0, 160.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="FAMILIE=PFAD.ts", help="mehrfach angebbar")
    ap.add_argument("--edf", type=Path, default=REPO / "examples/datasets/NiazyFMRI.edf")
    ap.add_argument("--out", type=Path, default=REPO / "output/pipeline_demo/run8_tuned/arms")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dc-mode", default="as_evaluated")
    args = ap.parse_args()

    import mne
    import torch

    import reference_chain
    from family_adapters import FamilyAdapter
    from facet.correction.deep_learning import DeepLearningCorrection

    args.out.mkdir(parents=True, exist_ok=True)
    for eintrag in args.arm:
        familie, _, pfad = eintrag.partition("=")
        ts = Path(pfad)
        if not ts.exists():
            print(f"  {familie}: {ts} fehlt -- uebersprungen")
            continue
        model_id = f"{familie}_deployment"

        class GitterAdapter(FamilyAdapter):
            def _load_model(self):
                if self._model is None:
                    m = torch.jit.load(str(ts), map_location=self.device)
                    m.eval()
                    self._model = m
                return self._model, torch

        t0 = time.perf_counter()
        adapter = GitterAdapter(model_id, device=args.device, dc_mode=args.dc_mode)
        pipe = reference_chain.build(args.edf,
                                     correctors=[DeepLearningCorrection(model=adapter)],
                                     include_pca=True, name=familie)
        res = pipe.run()
        if not res.success:
            print(f"  {familie}: Pipeline fehlgeschlagen -- {res.error}")
            continue
        raw = res.context.get_raw()
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
        sf = float(raw.info["sfreq"])
        i0, i1 = int(START_S * sf), int(STOP_S * sf)
        data = raw.get_data(picks=picks)[:, i0:i1].astype(np.float32) * 1e6
        namen = [raw.ch_names[i] for i in picks]
        trg = np.asarray(res.context.get_triggers() if res.context.has_triggers() else [],
                         dtype=np.int64)
        trg = trg[(trg >= i0) & (trg < i1)] - i0
        elapsed = time.perf_counter() - t0
        ziel = args.out / f"{familie}_tuned.npz"
        np.savez_compressed(ziel, data=data, ch_names=np.asarray(namen, dtype=object),
                            sfreq=sf, elapsed=elapsed, triggers=trg,
                            window_s=np.asarray([START_S, STOP_S], dtype=float))
        print(f"  {familie}: {elapsed:.0f} s -> {ziel}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
