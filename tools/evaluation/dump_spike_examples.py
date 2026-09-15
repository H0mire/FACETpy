"""Dump corrected traces for a deterministic set of spike examples.

Why a separate tool. The results protocol requires example figures with a
*pre-defined* selection and identical scaling
(``docs/research/results_evidence_pack_execution_plan.md`` §5.5.3, §5.6.3).
Picking examples after looking at the predictions is cherry-picking, and
re-deriving them inside a plotting script would mean re-running inference every
time a figure changes. This writes one NPZ holding the raw window, the true
clean, the spike mask and every arm's corrected trace for the first ``--n``
spike-bearing validation examples in ascending ``example_index`` order — a rule
fixed before any prediction is seen.

All arms are produced in a single pass over the dataset, because the NPZ is
several gigabytes and loading it once per model is the dominant cost.

The arm specification is JSON, so the exact set of compared models is itself a
citable artefact::

    [
      {"name": "cascade", "checkpoint": "...pt",
       "factory": "facet.models.demucs_mc.training:build_model",
       "kwargs": {"initial_channels": 32},
       "residual_mode": true, "max_channels": null},
      {"name": "demucs_direct", "checkpoint": "...pt", "factory": "...", "kwargs": {}}
    ]

Usage::

    .venv/bin/python tools/evaluation/dump_spike_examples.py \\
        --dataset output/weg_a_farm_v6_512/weg_a_spatiotemporal_dataset.npz \\
        --models-json output/results_evidence_pack/.../arms.json \\
        --n 6 --out output/model_evaluations/spike_example_traces.npz
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--models-json", type=Path, required=True)
    p.add_argument("--n", type=int, default=6, help="Number of spike examples, taken in ascending index order")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="mps")
    return p.parse_args()


def _resolve(spec: str):
    module_name, _, attr = spec.partition(":")
    return getattr(import_module(module_name), attr)


def _load_model(arm: dict, input_shape: tuple[int, int, int], device: str) -> torch.nn.Module:
    state = torch.load(arm["checkpoint"], map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    model = _resolve(arm["factory"])(input_shape=input_shape, **arm.get("kwargs", {}))
    model.load_state_dict(state)
    model = model.to(device).eval()
    export = getattr(model, "export_module", None)
    return export().to(device).eval() if callable(export) else model


def main() -> None:
    args = parse_args()
    arms = json.loads(args.models_json.read_text(encoding="utf-8"))
    if not arms:
        raise SystemExit("--models-json is empty")

    with np.load(args.dataset, allow_pickle=True) as b:
        split = b["example_split"]
        val = np.flatnonzero(split == 1)
        core = int(b["core_samples"][0])
        guard = int(b["guard_samples"][0])
        sl = slice(guard, guard + core)
        sfreq = float(b["sfreq"][0]) if "sfreq" in b.files else float("nan")

        spikes_all = b["spike_labels"][val][:, 0, sl]
        # Deterministic selection: the first n spike-bearing examples by index.
        # Fixed before any prediction exists, so the figure cannot be tuned.
        spike_rows = np.flatnonzero(spikes_all.any(axis=1))[: args.n]
        if spike_rows.size == 0:
            raise SystemExit("no spike-bearing validation example found")
        pick = val[spike_rows]

        clean = b["clean_center"][pick][:, 0, sl]
        artifact = b["artifact_center"][pick][:, 0, sl]
        template = b["artifact_center_template"][pick][:, 0, sl]
        spikes = b["spike_labels"][pick][:, 0, sl]
        # Contexts: only the selected rows, so this stays small.
        clean_ctx = b["clean_context"][pick][..., sl]
        artifact_ctx = b["artifact_context"][pick][..., sl]
        template_ctx = (
            b["artifact_context_template"][pick][..., sl]
            if "artifact_context_template" in b.files
            else None
        )

    noisy = clean + artifact
    corrected = {"aas_ideal": noisy - template, "null_output": np.zeros_like(clean)}
    meta_arms = []

    for arm in arms:
        residual = bool(arm.get("residual_mode", False))
        if residual and template_ctx is None:
            raise SystemExit(f"arm '{arm['name']}' needs residual_mode but the dataset has no context template")
        art = artifact_ctx - template_ctx if residual else artifact_ctx
        limit = arm.get("max_channels")
        if limit:
            art = art[:, :, : int(limit)]
            cln = clean_ctx[:, :, : int(limit)]
        else:
            cln = clean_ctx
        x = (cln + art).astype(np.float32)
        model = _load_model(arm, (x.shape[1], x.shape[2], core), args.device)
        with torch.no_grad():
            pred = model(torch.from_numpy(x).to(args.device))
            if isinstance(pred, dict):
                pred = pred[arm["output_key"]]
            pred = pred.cpu().numpy()[:, 0, :]
        base = corrected["aas_ideal"] if residual else noisy
        corrected[arm["name"]] = base - pred
        meta_arms.append({
            "name": arm["name"],
            "checkpoint": str(arm["checkpoint"]),
            "factory": arm["factory"],
            "kwargs": arm.get("kwargs", {}),
            "residual_mode": residual,
            "max_channels": limit,
            "input_shape": list(x.shape[1:]),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        example_index=spike_rows.astype(np.int64),
        dataset_row=pick.astype(np.int64),
        noisy=noisy.astype(np.float32),
        clean=clean.astype(np.float32),
        artifact=artifact.astype(np.float32),
        template=template.astype(np.float32),
        spikes=spikes.astype(np.float32),
        arm_names=np.array(list(corrected), dtype=object),
        **{f"corrected_{k}": v.astype(np.float32) for k, v in corrected.items()},
    )
    (args.out.with_suffix(".json")).write_text(
        json.dumps({
            "dataset": str(args.dataset),
            "sfreq_hz": sfreq,
            "core_samples": core,
            "selection_rule": "first n spike-bearing validation examples in ascending example_index order",
            "n_selected": int(spike_rows.size),
            "example_index": spike_rows.tolist(),
            "dataset_row": pick.tolist(),
            "arms": meta_arms,
            "reference_arms": {
                "aas_ideal": "noisy - artifact_center_template (perfect template recovery)",
                "null_output": "clean_hat = 0",
            },
        }, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.out} — {spike_rows.size} examples, arms: {', '.join(corrected)}")


if __name__ == "__main__":
    main()
