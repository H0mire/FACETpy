"""Sweep the RecoveredCleanLoss weights on one GPU (run_6).

The objective works — deletion is no longer reachable — but its two weights were
never tuned, and the single run overshot: spike recovery ratio 1.62 where 1.0 is
correct (``docs/research/run_6_results.md`` §3b). ``mse_weight`` is what anchors
absolute amplitude against the scale-invariant SI-SDR term, so it is the knob that
should pull the ratio down; ``spike_weight`` decides how much of the model's
capacity goes to the IED region.

Each configuration is a full ``facet-train`` run written to its own directory.
Configs are generated from the committed YAML so the sweep cannot silently drift
from the reproducible baseline.

Usage (on the GPU host)::

    .venv/bin/python tools/training/sweep_recovered_clean.py \
        --base-config configs/weg_a_demucs_mc_clean.yaml \
        --mse-weights 3 10 30 --spike-weights 5 20 \
        --max-epochs 60 --out-root sweeps/recovered_clean
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-config", type=Path, default=Path("configs/weg_a_demucs_mc_clean.yaml"))
    p.add_argument("--mse-weights", type=float, nargs="+", default=[3.0, 10.0, 30.0])
    p.add_argument("--spike-weights", type=float, nargs="+", default=[5.0, 20.0])
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64, help="A 5090 has room for far more than the MPS default")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-root", type=Path, default=Path("sweeps/recovered_clean"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    args.out_root.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.mse_weights, args.spike_weights))
    print(f"{len(combos)} configurations x {args.max_epochs} epochs on {args.device}\n", flush=True)

    results: list[dict] = []
    for i, (mse_w, spike_w) in enumerate(combos, start=1):
        tag = f"mse{mse_w:g}_spike{spike_w:g}"
        cfg = json.loads(json.dumps(base))          # deep copy
        cfg["model"]["device"] = args.device
        cfg["model"]["loss_kwargs"]["mse_weight"] = float(mse_w)
        cfg["model"]["loss_kwargs"]["spike_weight"] = float(spike_w)
        cfg["training"]["max_epochs"] = int(args.max_epochs)
        cfg["training"]["batch_size"] = int(args.batch_size)
        cfg["training"]["model_name"] = f"DemucsMCClean_{tag}"
        cfg["training"]["output_dir"] = str(args.out_root / tag)
        # T_max must follow max_epochs or the cosine schedule ends mid-run.
        sched = cfg["model"].get("scheduler_kwargs")
        if sched and "T_max" in sched:
            sched["T_max"] = int(args.max_epochs)

        cfg_path = args.out_root / f"{tag}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"[{i}/{len(combos)}] {tag}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(
            [sys.executable, "-m", "facet.training.cli", "fit", "--config", str(cfg_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}", flush=True)
        results.append({"tag": tag, "mse_weight": mse_w, "spike_weight": spike_w,
                        "returncode": proc.returncode, "config": str(cfg_path)})
        (args.out_root / "sweep_index.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\ndone — index at {args.out_root / 'sweep_index.json'}", flush=True)


if __name__ == "__main__":
    main()
