"""Create a compact overview of the recorded Run-7 (Phase-2) training histories."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "docs" / "research" / "run_7_logs"
OUT = (REPO / "output" / "thesis_results_by_phase" / "phase_2_pipeline_deployment"
       / "figure_phase2_training_curves.png")

# One recorded primary Run-7 deployment per family.  The seed-repeat and later
# context ablations are deliberately excluded: this figure documents the model
# configurations evaluated in the shared deployment comparison.
RUNS = (
    ("DPAE", "pod2_runs/dpaedeploymentniazyprooffit_20260826_195137/training.jsonl"),
    ("IC-U-Net", "pod1_runs/icunetdeploymentniazyprooffit_20260826_182808/training.jsonl"),
    ("Cascaded DAE", "pod2_runs/cascadeddaedeploymentniazyprooffit_20260826_195137/training.jsonl"),
    ("Nested GAN", "pod2_runs/nestedgandeploymentniazyprooffit_20260826_195137/training.jsonl"),
    ("DHCT-GAN", "pod2_runs/dhctgandeploymentniazyprooffit_20260826_195301/training.jsonl"),
    ("DHCT-GAN v2", "pod2_runs/dhctganv2deploymentniazyprooffit_20260826_195137/training.jsonl"),
    ("D4PM", "pod5_runs/d4pmdeploymentniazyprooffit_20260826_211657/training.jsonl"),
    ("DenoiseMamba", "pod1_runs/denoisemambadeploymentniazyprooffit_20260826_210335/training.jsonl"),
    ("Conv-TasNet", "pod1_runs/convtasnetdeploymentniazyprooffit_20260826_195217/training.jsonl"),
    ("Demucs", "pod1_runs/demucsdeploymentniazyprooffit_20260826_195217/training.jsonl"),
    ("SepFormer", "pod3_runs/sepformerdeploymentniazyprooffit_20260826_200317/training.jsonl"),
    ("Vision Transformer", "pod2_runs/vitspectrogramdeploymentniazyprooffit_20260826_192922/training.jsonl"),
    ("ST-GNN", "pod1_runs/stgnndeploymentniazyprooffit_20260826_195217/training.jsonl"),
    ("Cascaded Context DAE", "pod3_runs/cascadedcontextdaedeploymentniazyprooffit_20260826_200317/training.jsonl"),
)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 11), constrained_layout=True)
    for ax, (name, relative) in zip(axes.flat, RUNS):
        rows = read(LOGS / relative)
        epochs = [row["epoch"] for row in rows]
        for key, label, color in (("train_loss", "Training", "#2d6fae"),
                                  ("loss", "Training", "#2d6fae"),
                                  ("val_loss", "Validation", "#d95f02")):
            values = [row.get(key) for row in rows]
            if any(value is not None for value in values):
                ax.plot(epochs, values, lw=1.2, label=label, color=color)
                if key in {"train_loss", "loss"}:
                    break
        val = [row.get("val_loss") for row in rows]
        if any(value is not None for value in val):
            ax.plot(epochs, val, lw=1.2, label="Validation", color="#d95f02")
        ax.set_title(f"{name} ({len(rows)} epochs)", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.22)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=7, frameon=False, loc="best")
    for ax in axes.flat[len(RUNS):]:
        ax.set_axis_off()
    fig.suptitle("Phase 2: recorded training histories of Run-7 deployment models",
                 fontsize=14, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
