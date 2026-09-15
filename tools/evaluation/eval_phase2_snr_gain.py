#!/usr/bin/env python3
"""Evaluate selected Run-7 deployment exports on the Phase-1 unified holdout.

This creates a deliberately separate SNR-gain comparison: all final Phase-2
deployment editions are run on the same AAS-derived Niazy holdout used in
Phase 1.  It must not be confused with the real-pipeline residual table.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]

from evaluation.eval_unified_holdout import (  # noqa: E402
    DATASET_PATH, compute_holdout_indices, compute_metrics, load_holdout,
)
from pipeline_demo.family_adapters import (  # noqa: E402
    DEPLOYMENT_SPECS, load_model, predict_from_context,
)

OUT = ROOT / "output/thesis_results_by_phase/phase_2_pipeline_deployment"
TABLE = OUT / "table_phase2_unified_holdout_snr_gain.csv"

# The exact primary deployment exports that produced the shared Phase-2 table.
# Explicit paths avoid accidentally selecting a later seed or context ablation.
EXPORTS = {
    "DPAE": "training_output/dpaedeploymentniazyprooffit_20260826_195137/exports/dpae_deployment.ts",
    "IC-U-Net": "training_output/icunetdeploymentniazyprooffit_20260826_200640/exports/ic_unet_deployment.ts",
    "Cascaded DAE": "training_output/cascadeddaedeploymentniazyprooffit_20260826_195137/exports/cascaded_dae_deployment.ts",
    "Nested GAN": "training_output/nestedgandeploymentniazyprooffit_20260826_195137/exports/nested_gan_deployment.ts",
    "DHCT-GAN": "training_output/dhctgandeploymentniazyprooffit_20260826_195301/exports/dhct_gan_deployment.ts",
    "DHCT-GAN v2": "training_output/dhctganv2deploymentniazyprooffit_20260826_195137/exports/dhct_gan_v2_deployment.ts",
    "DenoiseMamba": "training_output/denoisemambadeploymentniazyprooffit_20260826_210335/exports/denoise_mamba_deployment_cpu.ts",
    "Conv-TasNet": "training_output/convtasnetdeploymentniazyprooffit_20260826_195217/exports/conv_tasnet_deployment.ts",
    "Demucs": "training_output/demucsdeploymentniazyprooffit_20260826_195217/exports/demucs_deployment_cpu.ts",
    "SepFormer": "training_output/sepformerdeploymentniazyprooffit_20260826_200317/exports/sepformer_deployment.ts",
    "Vision Transformer": "training_output/vitspectrogramdeploymentniazyprooffit_20260826_192922/exports/vit_spectrogram_deployment.ts",
    "ST-GNN": "training_output/stgnndeploymentniazyprooffit_20260826_195217/exports/st_gnn_deployment.ts",
    "Cascaded Context DAE": "training_output/cascadedcontextdaedeploymentniazyprooffit_20260826_200317/exports/cascaded_context_dae_deployment.ts",
}

# Kept identical to the Phase-1 unified-holdout ranking.
FAMILY = {
    "Demucs": "Audio", "Conv-TasNet": "Audio", "SepFormer": "Audio",
    "Nested GAN": "GAN", "DHCT-GAN": "GAN", "DHCT-GAN v2": "GAN",
    "DPAE": "Discriminative", "IC-U-Net": "Discriminative",
    "Cascaded DAE": "Autoencoder", "Cascaded Context DAE": "Autoencoder",
    "DenoiseMamba": "SSM", "Vision Transformer": "Vision", "ST-GNN": "Graph",
}
FAMILY_COLOR = {
    "Audio": "#1f77b4", "Discriminative": "#2ca02c", "Autoencoder": "#17becf",
    "SSM": "#ff7f0e", "Vision": "#9467bd", "Graph": "#8c564b", "GAN": "#d62728",
}
DISPLAY_NAME = {
    "Nested GAN": "Nested-GAN", "DenoiseMamba": "Denoise-Mamba",
    "Vision Transformer": "ViT Spectrogram", "DHCT-GAN": "DHCT-GAN v1",
}


def main() -> None:
    import torch

    device = "cpu"
    ds = load_holdout(DATASET_PATH, compute_holdout_indices())
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    if TABLE.exists():
        with TABLE.open(newline="") as f:
            rows = list(csv.DictReader(f))
    completed = {str(row["model"]) for row in rows}
    for name, rel_path in EXPORTS.items():
        if name in completed:
            print(f"cached {name}", flush=True)
            continue
        filename = Path(rel_path).name
        model_id = filename.removesuffix("_deployment_cpu.ts").removesuffix("_deployment.ts")
        deployment_id = f"{model_id}_deployment"
        if deployment_id not in DEPLOYMENT_SPECS:
            raise KeyError(f"No deployment specification for {name}: {deployment_id}")
        path = ROOT / rel_path
        if not path.exists():
            print(f"skip {name}: missing {path}")
            continue
        print(f"evaluating {name}", flush=True)
        model = torch.jit.load(str(path), map_location=device).eval()
        pred = predict_from_context(DEPLOYMENT_SPECS[deployment_id], model, ds["noisy_context"], device=device)
        metrics = compute_metrics(ds["noisy_center"], ds["clean_center"], ds["artifact_center"], pred, sfreq_hz=ds["sfreq"])
        rows.append({"model": name, "snr_gain_db": float(metrics["clean_snr_improvement_db"])})
        rows.sort(key=lambda row: float(row["snr_gain_db"]), reverse=True)
        with TABLE.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "snr_gain_db"])
            writer.writeheader(); writer.writerows(rows)
        del model

    rows.sort(key=lambda row: float(row["snr_gain_db"]), reverse=True)
    with TABLE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "snr_gain_db"])
        writer.writeheader(); writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    model_names = [str(row["model"]) for row in rows][::-1]
    labels = [DISPLAY_NAME.get(name, name) for name in model_names]
    values = [float(row["snr_gain_db"]) for row in rows][::-1]
    colors = [FAMILY_COLOR[FAMILY[name]] for name in model_names]
    bars = ax.barh(labels, values, color=colors, alpha=0.85)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Clean-signal SNR improvement (dB; higher is better)")
    ax.set_title("Model ranking on the unified holdout", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    for bar, value in zip(bars, values):
        ax.text(value + (0.15 if value >= 0 else -0.15), bar.get_y() + bar.get_height()/2,
                f"{value:.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    from matplotlib.patches import Patch
    used_families = [family for family in FAMILY_COLOR if family in set(FAMILY.values())]
    ax.legend(handles=[Patch(color=FAMILY_COLOR[family], label=family) for family in used_families],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Model family",
              frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figure_phase2_unified_holdout_snr_gain.png", dpi=220, bbox_inches="tight")
    print("done", flush=True)


if __name__ == "__main__":
    main()
