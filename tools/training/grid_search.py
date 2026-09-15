"""Grid search over the cascade formulation (run_6 §3c).

Sweeping only the two loss weights was not enough to tell whether the cascade
helps: correlation with the true clean is the metric that matters and it barely
moved. This grids the axes that plausibly control it — how hard the amplitude is
anchored, how much capacity goes to the IED region, the learning rate, and model
width — and scores every run with the *same* comparison the report uses, so the
result is directly readable against FARM instead of against a training loss.

Each run writes its own directory; ``grid_results.json`` accumulates the scored
rows so a partial sweep is still usable if the host goes away.

Usage (on a GPU host)::

    .venv/bin/python tools/training/grid_search.py \
        --base-config configs/weg_a_demucs_mc_cascade.yaml \
        --dataset output/weg_a_farm_v6_512/weg_a_spatiotemporal_dataset.npz \
        --shard 0 --n-shards 2 --out-root grids/cascade
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-config", type=Path, default=Path("configs/weg_a_demucs_mc_cascade.yaml"))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--mse-weights", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    p.add_argument("--spike-weights", type=float, nargs="+", default=[1.0, 20.0])
    p.add_argument("--learning-rates", type=float, nargs="+", default=[3e-4, 1e-3])
    p.add_argument("--initial-channels", type=int, nargs="+", default=[32, 64])
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-root", type=Path, default=Path("grids/cascade"))
    p.add_argument("--shard", type=int, default=0, help="This host's slice of the grid")
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scoring — the same comparison the report makes, not the training loss
# ---------------------------------------------------------------------------


def load_val(dataset: Path) -> dict:
    with np.load(dataset, allow_pickle=True) as b:
        v = np.flatnonzero(b["example_split"] == 1)
        core, guard = int(b["core_samples"][0]), int(b["guard_samples"][0])
        sl = slice(guard, guard + core)
        clean = b["clean_center"][v][:, 0, sl].astype(np.float64)
        art = b["artifact_center"][v][:, 0, sl].astype(np.float64)
        tmpl = b["artifact_center_template"][v][:, 0, sl].astype(np.float64)
        tmpl_ctx = b["artifact_context_template"][v][..., sl]
        ctx = (b["clean_context"][v] + b["artifact_context"][v])[..., sl]
        spk = b["spike_labels"][v][:, 0, sl] > 0
    return {"clean": clean, "art": art, "tmpl": tmpl, "spk": spk, "core": core,
            # residual-mode input: the model sees what FARM leaves, not the raw signal
            "ctx_resid": (ctx - tmpl_ctx).astype(np.float32)}


def score(pred_residual: np.ndarray, d: dict) -> dict:
    """Model output is the residual FARM leaves, so clean_hat = (noisy - template) - pred."""
    clean, art, tmpl, spk = d["clean"], d["art"], d["tmpl"], d["spk"]
    clean_hat = (clean + art - tmpl) - pred_residual
    rms = lambda x: float(np.sqrt(np.mean(x**2))) * 1e6  # noqa: E731
    ratios = []
    for i in np.flatnonzero(spk.any(1)):
        p = int(np.argmax(np.abs(clean[i] * spk[i])))
        if abs(clean[i, p]) > 0:
            ratios.append(abs(clean_hat[i, p]) / abs(clean[i, p]))
    return {
        "err_uv": rms(clean_hat - clean),
        "corr_clean": float(np.corrcoef(clean_hat.ravel(), clean.ravel())[0, 1]),
        "spike_ratio": float(np.median(ratios)) if ratios else float("nan"),
    }


def farm_reference(d: dict) -> dict:
    """FARM alone = the cascade's model contributes nothing on top of the template."""
    return score(np.zeros_like(d["art"]), d)


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    args.out_root.mkdir(parents=True, exist_ok=True)
    d = load_val(args.dataset)

    grid = list(itertools.product(args.mse_weights, args.spike_weights,
                                  args.learning_rates, args.initial_channels))
    mine = [g for i, g in enumerate(grid) if i % args.n_shards == args.shard]
    ref = farm_reference(d)
    print(f"grid {len(grid)} total, shard {args.shard}/{args.n_shards} -> {len(mine)} runs")
    print(f"FARM reference: err {ref['err_uv']:.1f} uV  corr {ref['corr_clean']:.3f}  ratio {ref['spike_ratio']:.2f}\n",
          flush=True)

    from facet.models.demucs_mc.training import build_model

    rows: list[dict] = []
    for i, (mse_w, spike_w, lr, ch) in enumerate(mine, start=1):
        tag = f"mse{mse_w:g}_spk{spike_w:g}_lr{lr:g}_ch{ch}"
        cfg = json.loads(json.dumps(base))
        cfg["model"]["device"] = args.device
        cfg["model"]["kwargs"]["initial_channels"] = int(ch)
        cfg["model"]["loss_kwargs"]["mse_weight"] = float(mse_w)
        cfg["model"]["loss_kwargs"]["spike_weight"] = float(spike_w)
        cfg["data"]["kwargs"]["path"] = str(args.dataset)
        cfg["training"].update(max_epochs=int(args.max_epochs), batch_size=int(args.batch_size),
                               learning_rate=float(lr), model_name=f"Cascade_{tag}",
                               output_dir=str(args.out_root / tag))
        sched = cfg["model"].get("scheduler_kwargs") or {}
        if "T_max" in sched:
            sched["T_max"] = int(args.max_epochs)

        cfg_path = args.out_root / f"{tag}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"[{i}/{len(mine)}] {tag}", flush=True)
        if args.dry_run:
            continue

        proc = subprocess.run([sys.executable, "-m", "facet.training.cli", "fit", "--config", str(cfg_path)],
                              capture_output=True, text=True)
        row = {"tag": tag, "mse_weight": mse_w, "spike_weight": spike_w,
               "learning_rate": lr, "initial_channels": ch, "returncode": proc.returncode}
        cks = sorted(glob.glob(str(args.out_root / tag / "*" / "checkpoints" / "epoch*.pt")))
        if proc.returncode != 0 or not cks:
            row["error"] = (proc.stderr or proc.stdout)[-400:]
            print(f"    failed: {row['error'][:200]}", flush=True)
        else:
            state = torch.load(cks[-1], map_location="cpu", weights_only=False)
            for k in ("model_state_dict", "state_dict", "model"):
                if isinstance(state, dict) and k in state and isinstance(state[k], dict):
                    state = state[k]
                    break
            # Build from the config's own kwargs, not defaults: a factory default
            # that differs from the trained architecture (attention_levels was the
            # one that bit) makes the checkpoint silently unloadable.
            model = build_model(
                input_shape=(d["ctx_resid"].shape[1], d["ctx_resid"].shape[2], d["core"]),
                **cfg["model"]["kwargs"],
            ).to(args.device).eval()
            model.load_state_dict(state)
            pred = np.empty_like(d["art"])
            with torch.no_grad():
                for s in range(0, len(d["ctx_resid"]), 128):
                    batch = torch.from_numpy(d["ctx_resid"][s:s + 128]).to(args.device)
                    pred[s:s + 128] = model(batch).cpu().numpy()[:, 0, :]
            row |= score(pred, d)
            print(f"    err {row['err_uv']:.1f} uV | corr {row['corr_clean']:.3f} | ratio {row['spike_ratio']:.2f}",
                  flush=True)
        rows.append(row)
        (args.out_root / f"grid_results_shard{args.shard}.json").write_text(
            json.dumps({"farm_reference": ref, "rows": rows}, indent=2), encoding="utf-8")

    ok = [r for r in rows if "corr_clean" in r]
    if ok:
        best = max(ok, key=lambda r: r["corr_clean"])
        print(f"\nbest by corr_clean: {best['tag']} -> corr {best['corr_clean']:.3f} "
              f"(FARM {ref['corr_clean']:.3f}), err {best['err_uv']:.1f} uV, ratio {best['spike_ratio']:.2f}")


if __name__ == "__main__":
    main()
