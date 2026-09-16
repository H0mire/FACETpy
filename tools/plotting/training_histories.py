"""Plot the recorded histories for one thesis phase without selecting new runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from masterthesis_guide.reproduce import (  # noqa: E402 - repository path is set before checkout-only imports
    load_catalog,
    sha256,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    catalog = load_catalog()
    histories = []
    for eid, experiment in catalog["experiments"].items():
        selected = eid.startswith("holdout_") if args.phase == 1 else eid.startswith("training_run7_")
        if not selected:
            continue
        logs = [ROOT / p for p in experiment["evidence"] if p.endswith("training.jsonl")]
        if len(logs) > 1:
            raise ValueError(f"Ambiguous training history: {eid}")
        if logs:
            histories.append((eid, logs[0]))
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 11), constrained_layout=True)
    for ax, (eid, path) in zip(axes.flat, histories, strict=False):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        epochs = [r["epoch"] for r in rows]
        for keys, label in [(("train_loss", "loss"), "Training"), (("val_loss",), "Validation")]:
            key = next((k for k in keys if any(r.get(k) is not None for r in rows)), None)
            if key:
                ax.plot(epochs, [r.get(key) for r in rows], label=label, lw=1.1)
        ax.set_title(eid.removeprefix("holdout_").removeprefix("training_run7_"), fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(histories) :]:
        ax.set_axis_off()
    fig.suptitle(f"Phase {args.phase}: recorded training histories")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps({eid: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for eid, p in histories}, indent=2)
    )


if __name__ == "__main__":
    main()
