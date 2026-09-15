"""Compare the recorded Phase-1 and Run-7 DenoiseMamba training histories."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
PHASE1 = REPO / "training_output/denoisemambaniazyprooffit_20260510_193847/training.jsonl"
PHASE2 = REPO / "docs/research/run_7_logs/pod1_runs/denoisemambadeploymentniazyprooffit_20260826_210335/training.jsonl"
OUT = REPO / "output/thesis_results_by_phase/phase_2_pipeline_deployment/figure_denoisemamba_training_curves.png"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_axis(ax, history: list[dict], title: str) -> None:
    epoch = [row["epoch"] for row in history]
    ax.plot(epoch, [row["train_loss"] for row in history], marker="o", ms=3,
            lw=1.5, label="Training loss")
    ax.plot(epoch, [row["val_loss"] for row in history], marker="o", ms=3,
            lw=1.5, label="Validation loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def main() -> None:
    p1, p2 = load(PHASE1), load(PHASE2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    plot_axis(axes[0], p1, "Phase 1: original DenoiseMamba (11 epochs)")
    plot_axis(axes[1], p2, "Phase 2 / Run 7: deployment DenoiseMamba (35 epochs)")
    fig.suptitle("Recorded DenoiseMamba training histories", fontsize=13, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
