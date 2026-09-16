"""Carve a locked holdout out of a dataset's validation split.

A configuration selected on the validation split and then reported on that same
split reports a number that selection already optimised. The fix is a third
partition that neither training nor early stopping ever sees.

``example_split`` uses 0 for train and 1 for validation. This tool relabels the
**later** centre epochs of the validation region to **2**:

* the trainer's ``train_val_split`` selects ``== 0`` and ``== 1``, so examples
  marked 2 enter neither the gradient nor the early-stopping signal;
* the evaluation tool takes ``--split 2`` to report on them.

The cut is by **centre epoch**, not by example, so no epoch's electrode
replicates end up on both sides, and a guard band of ``context_epochs`` epochs is
dropped at the seam because neighbouring centre epochs share context.

Usage::

    .venv/bin/python tools/dataset_building/lock_holdout_split.py \\
        --dataset output/weg_a_farm_v10_locked_512/weg_a_spatiotemporal_dataset.npz \\
        --locked-fraction 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument(
        "--locked-fraction",
        type=float,
        default=0.5,
        help="Share of the validation EPOCHS (the later ones) that become locked.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with np.load(args.dataset, allow_pickle=True) as b:
        data = {k: b[k] for k in b.files}

    split = data["example_split"].astype(np.int64)
    epoch = data["center_epoch_index"].astype(np.int64)
    context = int(data["context_epochs"][0])
    if (split == 2).any():
        raise SystemExit("dataset already carries a locked split (example_split == 2)")

    val_epochs = np.array(sorted({int(e) for e in epoch[split == 1]}))
    if val_epochs.size < 4:
        raise SystemExit(f"validation split has only {val_epochs.size} epochs — nothing to carve")

    cut = int(round(val_epochs.size * (1.0 - args.locked_fraction)))
    select_epochs = set(val_epochs[:cut].tolist())
    locked_epochs = set(val_epochs[cut:].tolist())

    # Guard band: neighbouring centre epochs share context_epochs - 1 input
    # epochs, so the epochs straddling the seam belong to neither side.
    seam_lo, seam_hi = val_epochs[cut - 1], val_epochs[cut]
    guard = {e for e in val_epochs if seam_lo - context < e <= seam_hi + context - 1}
    select_epochs -= guard
    locked_epochs -= guard

    new_split = split.copy()
    dropped = 0
    for i, (sp, ep) in enumerate(zip(split, epoch, strict=False)):
        if sp != 1:
            continue
        e = int(ep)
        if e in locked_epochs:
            new_split[i] = 2
        elif e in select_epochs:
            new_split[i] = 1
        else:
            new_split[i] = -1  # guard band, excluded everywhere
            dropped += 1

    report = {
        "dataset": str(args.dataset),
        "context_epochs": context,
        "n_val_epochs_before": int(val_epochs.size),
        "n_selection_epochs": len(select_epochs),
        "n_locked_epochs": len(locked_epochs),
        "n_guard_epochs_dropped": len(guard),
        "n_examples_train": int((new_split == 0).sum()),
        "n_examples_selection": int((new_split == 1).sum()),
        "n_examples_locked": int((new_split == 2).sum()),
        "n_examples_dropped_at_seam": dropped,
        "selection_epoch_range": [int(min(select_epochs)), int(max(select_epochs))] if select_epochs else [],
        "locked_epoch_range": [int(min(locked_epochs)), int(max(locked_epochs))] if locked_epochs else [],
        "epoch_overlap": sorted(select_epochs & locked_epochs),
        "note": "example_split: 0 train, 1 selection/early stopping, 2 locked holdout, "
        "-1 guard band at the seam (used nowhere).",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["epoch_overlap"]:
        raise SystemExit("selection and locked splits overlap — refusing to write")
    if args.dry_run:
        return

    data["example_split"] = new_split
    np.savez_compressed(args.dataset, **data)
    meta_path = args.dataset.with_name(args.dataset.stem + "_metadata.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["locked_split"] = report
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.dataset}")


if __name__ == "__main__":
    main()
