"""Write a channel-reduced copy of a Weg-A spatio-temporal dataset.

The full 7-electrode dataset is 8.1 GB, which is fine on local disk and painful
over a slow uplink to a GPU host. The paper configuration of a single-channel
model only ever reads the target electrode, so shipping seven of them is pure
transfer cost.

This keeps the first ``--channels`` electrodes of every context array. The builder
writes the target electrode at index 0 and its geodesic neighbours after it, so a
prefix is always "target plus the nearest N-1" and never silently retargets the
model — the same invariant :class:`NPZSpatioTemporalDataset`'s ``max_channels``
relies on, and it is asserted here rather than assumed.

Everything else — split, spike labels, centre targets, templates, provenance — is
copied unchanged, so the subset trains and evaluates exactly like the original at
that channel count.

Usage::

    .venv/bin/python tools/dataset_building/export_channel_subset.py \\
        --input  output/weg_a_farm_v7_k6_512/weg_a_spatiotemporal_dataset.npz \\
        --output output/weg_a_farm_v7_k1_512/weg_a_spatiotemporal_dataset.npz \\
        --channels 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: Arrays with an electrode axis at position 2 (examples, epochs, channels, samples).
CONTEXT_KEYS = ("clean_context", "artifact_context", "artifact_context_template")
#: Arrays with an electrode axis at position 1 (examples, channels).
CHANNEL_INDEX_KEYS = ("neighbor_channel_indices",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--channels", type=int, required=True, help="Electrodes to keep, counting from the target")
    p.add_argument(
        "--drop-context-template",
        action="store_true",
        help="Omit artifact_context_template. It is only read in residual/cascade mode; the centre "
        "template (needed for the FARM reference in evaluation) is always kept.",
    )
    p.add_argument("--compress", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with np.load(args.input, allow_pickle=True) as bundle:
        keys = list(bundle.files)
        n_channels = int(bundle["clean_context"].shape[2])
        keep = max(1, min(int(args.channels), n_channels))

        # The prefix-is-target invariant, checked rather than trusted.
        neighbours = bundle["neighbor_channel_indices"]
        target = bundle["target_channel_index"]
        if not bool((neighbours[:, 0] == target).all()):
            raise SystemExit(
                "neighbor_channel_indices[:, 0] is not the target channel for every example; "
                "a channel prefix would retarget the model. Refusing to write a subset."
            )

        out: dict[str, np.ndarray] = {}
        for key in keys:
            if key in CONTEXT_KEYS:
                if key == "artifact_context_template" and args.drop_context_template:
                    continue
                out[key] = bundle[key][:, :, :keep]
            elif key in CHANNEL_INDEX_KEYS:
                out[key] = bundle[key][:, :keep]
            else:
                out[key] = bundle[key]
        out["k_neighbors"] = np.asarray([keep - 1])

    saver = np.savez_compressed if args.compress else np.savez
    saver(args.output, **out)

    size_in = args.input.stat().st_size
    size_out = args.output.stat().st_size
    print(f"in  {args.input}  {size_in / 1e9:.2f} GB  ({n_channels} electrodes)")
    print(
        f"out {args.output}  {size_out / 1e9:.2f} GB  ({keep} electrodes)"
        f"{'  [context template dropped]' if args.drop_context_template else ''}"
    )
    print(f"    {size_in / max(size_out, 1):.1f}x smaller")

    meta_in = args.input.with_name(args.input.stem + "_metadata.json")
    if meta_in.exists():
        meta = json.loads(meta_in.read_text(encoding="utf-8"))
        meta |= {
            "derived_from": str(args.input),
            "k_neighbors": keep - 1,
            "input_shape": [
                meta.get("context_epochs", 7),
                keep,
                meta.get("core_samples", 512) + 2 * meta.get("guard_samples", 32),
            ],
            "note": "Channel subset written by tools/dataset_building/export_channel_subset.py. "
            "Electrode 0 is the target channel; the rest are its nearest geodesic neighbours.",
        }
        meta_out = args.output.with_name(args.output.stem + "_metadata.json")
        meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"    metadata -> {meta_out}")


if __name__ == "__main__":
    main()
