"""Create a compact overview of the recorded Phase-1 model training curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
EVALUATIONS = REPO / "output/model_evaluations"
OUT = REPO / "output/thesis_results_by_phase/phase_1_unified_holdout/figure_phase1_training_curves.png"


def histories() -> list[tuple[str, list[dict]]]:
    collected: list[tuple[str, list[dict]]] = []
    for manifest in sorted(EVALUATIONS.glob("*/holdout_v1/evaluation_manifest.json")):
        data = json.loads(manifest.read_text())
        checkpoint = data.get("config", {}).get("checkpoint")
        if not checkpoint:
            continue  # AAS baselines are not trained neural models.
        # Evaluation manifests preserve the host path used at evaluation time.
        # Only the training-run directory name is portable across checkouts.
        run_name = Path(checkpoint).parent.parent.name
        log = REPO / "training_output" / run_name / "training.jsonl"
        if not log.exists():
            continue
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        collected.append((data["model_name"], rows))
    return collected


def values(rows: list[dict], key: str) -> list[float] | None:
    if not all(key in row for row in rows):
        return None
    return [row[key] for row in rows]


def main() -> None:
    all_histories = histories()
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 11), constrained_layout=True)
    for ax, (name, rows) in zip(axes.flat, all_histories):
        epochs = [row["epoch"] for row in rows]
        train = values(rows, "train_loss") or values(rows, "loss")
        val = values(rows, "val_loss")
        if train is not None:
            ax.plot(epochs, train, lw=1.25, label="Training")
        if val is not None:
            ax.plot(epochs, val, lw=1.25, label="Validation")
        ax.set_title(f"{name} ({len(rows)} epochs)", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.22)
        if train is not None or val is not None:
            ax.legend(fontsize=7, frameon=False, loc="best")
    for ax in axes.flat[len(all_histories):]:
        ax.set_axis_off()
    fig.suptitle("Phase 1: recorded training histories of evaluated neural models", fontsize=14,
                 fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
