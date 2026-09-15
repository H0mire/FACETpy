"""Assert each family adapter reproduces the holdout inference exactly.

The adapters re-express fourteen input contracts in one parametrised function.
Re-expressing a contract is exactly the kind of change that looks right and is
wrong by a demean axis or a slice offset — and the failure is silent, because a
mis-packed input still produces a plausible-looking artifact.

So the packing is not reviewed, it is tested: for every model, run
``predict_from_context`` on the holdout context and compare against that model's
own inference function from ``tools/evaluation/eval_unified_holdout.py``. Same data, same
checkpoint, same device. Any mismatch above floating-point noise is a defect in
the adapter, not a tolerance to widen.

Usage::

    .venv/bin/python tools/pipeline_demo/verify_family_adapters.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

from evaluation.eval_unified_holdout import (                                    # noqa: E402
    INFERENCE_FUNCS, MODELS, TRAIN_ROOT, _load_torchscript, _resolve_ts_path,
    compute_holdout_indices, load_holdout,
)
from pipeline_demo.family_adapters import (                          # noqa: E402
    FAMILY_SPECS, load_model, predict_from_context,
)

#: Relative tolerance on the RMS of the difference, against the RMS of the
#: reference. Not zero: batching order and float32 accumulation differ slightly.
RTOL = 1e-4


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--n-windows", type=int, default=24,
                   help="Prefix of the holdout used for the check; the contract is "
                        "shape-and-axis logic, so a prefix settles it.")
    p.add_argument("--out", type=Path, default=REPO / "output/model_evaluations/family_adapters")
    p.add_argument("--skip", nargs="*", default=["d4pm"],
                   help="Models too slow to verify on CPU by default.")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    idx = compute_holdout_indices(n=833)[:args.n_windows]
    ds = load_holdout(REPO / "output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz", idx)
    ctx = ds["noisy_context"]                       # (N, T, C, S)

    rows = []
    chosen = args.models or [m for m in FAMILY_SPECS if m not in set(args.skip)]
    for model_id in chosen:
        packing = FAMILY_SPECS[model_id]
        # Resolve the glob BEFORE handing the spec to the reference function.
        # eval_unified_holdout.run_model does this; the inference functions
        # themselves use spec.ts_path verbatim, so an unresolved pattern reaches
        # torch.jit.load and fails there. Missed once already.
        base = MODELS[model_id]
        resolved = _resolve_ts_path(base.ts_path)
        spec = type(base)(**{**base.__dict__, "ts_path": resolved}) if resolved != base.ts_path else base
        try:
            reference = INFERENCE_FUNCS[model_id](spec, ds, device=args.device)
            model = load_model(model_id, args.device)
            mine = predict_from_context(packing, model, ctx, device=args.device)
        except Exception as exc:                              # noqa: BLE001
            rows.append({"model_id": model_id, "family": packing.family,
                         "status": f"FEHLER: {str(exc).splitlines()[-1][:160]}",
                         "matches": False})
            print(f"  [Fehler]  {model_id:22s} {rows[-1]['status']}", flush=True)
            continue
        if mine.shape != reference.shape:
            rows.append({"model_id": model_id, "family": packing.family,
                         "status": f"Form {mine.shape} != {reference.shape}", "matches": False})
            print(f"  [Form]    {model_id:22s} {rows[-1]['status']}", flush=True)
            continue
        ref_rms = float(np.sqrt(np.mean(reference ** 2)))
        diff_rms = float(np.sqrt(np.mean((mine - reference) ** 2)))
        rel = diff_rms / max(ref_rms, 1e-30)
        ok = rel <= RTOL
        rows.append({
            "model_id": model_id, "family": packing.family, "packing": packing.packing,
            "demean": packing.demean, "output": packing.output,
            "reference_rms": ref_rms, "difference_rms": diff_rms,
            "relative_difference": rel, "matches": ok,
            "status": "identisch" if ok else "ABWEICHUNG",
        })
        print(f"  [{'ok' if ok else 'ABWEICHUNG':^9}] {model_id:22s} "
              f"rel {rel:.3e}   (ref RMS {ref_rms:.3e})", flush=True)

    n_ok = sum(1 for r in rows if r["matches"])
    (args.out / "adapter_verification.json").write_text(json.dumps({
        "n_windows": int(len(idx)), "device": args.device, "rtol": RTOL,
        "n_checked": len(rows), "n_matching": n_ok,
        "skipped": args.skip,
        "meaning": "Jeder Adapter wird gegen die Inferenzfunktion desselben Modells aus "
                   "tools/evaluation/eval_unified_holdout.py geprüft — gleiche Daten, gleicher "
                   "Checkpoint, gleiches Gerät. Abweichung = Defekt im Adapter.",
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{n_ok}/{len(rows)} Adapter reproduzieren die Referenzinferenz "
          f"-> {args.out / 'adapter_verification.json'}")
    if n_ok != len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
