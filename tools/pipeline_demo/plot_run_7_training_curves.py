"""Create thesis-ready overviews from the recovered Run-7 training logs.

The script only reads the saved JSONL logs. It does not load checkpoints, start
training, or require a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Curve:
    run_name: str
    group_key: str
    group_label: str
    completed: bool
    epochs: tuple[int, ...]
    train_loss: tuple[float, ...]
    val_loss: tuple[float, ...]


GROUPS: tuple[tuple[str, str, str], ...] = (
    ("cascadedcontextdae", "cascaded_context_dae", "Cascaded context DAE"),
    ("cascadeddae", "cascaded_dae", "Cascaded DAE"),
    ("convtasnet", "conv_tasnet", "Conv-TasNet"),
    ("d4pm", "d4pm", "D4PM"),
    ("demucs", "demucs", "Demucs"),
    ("denoisemamba", "denoise_mamba", "DenoiseMamba"),
    ("dhctganstrict", "dhct_gan_strict", "DHCT-GAN strict"),
    ("dhctganv2", "dhct_gan_v2", "DHCT-GAN v2"),
    ("dhctgan", "dhct_gan", "DHCT-GAN"),
    ("dpae", "dpae", "DPAE"),
    ("icunet", "ic_unet", "IC-U-Net"),
    ("nestedgan", "nested_gan", "Nested-GAN"),
    ("sepformer", "sepformer", "SepFormer"),
    ("stgnn", "st_gnn", "ST-GNN"),
    ("vitspectrogram", "vit_spectrogram", "ViT-Spectrogram"),
)


def _group_for(run_name: str) -> tuple[str, str]:
    normalized = run_name.lower().replace("_", "")
    for prefix, key, label in GROUPS:
        if normalized.startswith(prefix):
            return key, label
    raise ValueError(f"Unknown Run-7 training family: {run_name}")


def load_curves(log_root: Path) -> list[Curve]:
    curves: list[Curve] = []
    for path in sorted(log_root.glob("pod*_runs/*/training.jsonl")):
        records = []
        for raw_line in path.read_text().splitlines():
            if not raw_line.strip():
                continue
            records.append(json.loads(raw_line))
        if not records:
            continue

        run_name = path.parent.name
        group_key, group_label = _group_for(run_name)
        epochs = tuple(int(row["epoch"]) for row in records)
        train_loss = tuple(float(row["train_loss"]) for row in records)
        val_loss = tuple(float(row["val_loss"]) for row in records)
        curves.append(
            Curve(
                run_name=run_name,
                group_key=group_key,
                group_label=group_label,
                completed=(path.parent / "summary.json").exists(),
                epochs=epochs,
                train_loss=train_loss,
                val_loss=val_loss,
            )
        )
    return curves


def _style_axis(axis: plt.Axes) -> None:
    axis.axhline(0.0, color="#8d99a6", linewidth=0.7, zorder=0)
    axis.grid(True, color="#dfe5eb", linewidth=0.55, alpha=0.8)
    axis.set_yscale("symlog", linthresh=0.05, linscale=0.8)
    axis.tick_params(axis="both", labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#aeb9c4")
        spine.set_linewidth(0.8)


def _is_main_deployment_run(curve: Curve) -> bool:
    normalized = curve.run_name.lower().replace("_", "")
    return "niazyprooffit" in normalized and "ctx" not in normalized and "long" not in normalized


def _timestamp_key(curve: Curve) -> str:
    match = re.search(r"(\d{8}_\d{6})$", curve.run_name)
    return match.group(1) if match else ""


def _tail_mad(curve: Curve) -> float:
    tail = curve.val_loss[-min(20, len(curve.val_loss)) :]
    center = statistics.median(tail)
    return statistics.median(abs(value - center) for value in tail)


def select_stable_curve_per_family(curves: list[Curve]) -> list[Curve]:
    selected: list[Curve] = []
    for _, group_key, _ in GROUPS:
        # The strict DHCT-GAN runs were a later objective audit, not one of the
        # fourteen proof-fit architecture families.
        if group_key == "dhct_gan_strict":
            continue
        candidates = [
            curve
            for curve in curves
            if curve.group_key == group_key
            and curve.completed
            and _is_main_deployment_run(curve)
        ]
        if not candidates:
            raise RuntimeError(f"No completed base deployment run for {group_key}")

        # Prefer the lowest tail variation within the family. Recency is only
        # the tie-breaker, so an incomplete or visibly unstable latest run does
        # not replace a better completed curve.
        candidates.sort(key=_timestamp_key, reverse=True)
        selected.append(min(candidates, key=_tail_mad))
    return selected


def plot_stable_curve_per_family(curves: list[Curve], output: Path) -> list[Curve]:
    selected = select_stable_curve_per_family(curves)
    fig, axes = plt.subplots(5, 3, figsize=(14, 17), dpi=180)

    for axis, curve in zip(axes.flat, selected, strict=False):
        axis.plot(
            curve.epochs,
            curve.train_loss,
            color="#b7652a",
            linewidth=1.25,
            linestyle="--",
            alpha=0.90,
        )
        axis.plot(
            curve.epochs,
            curve.val_loss,
            color="#315a91",
            linewidth=1.35,
            alpha=0.95,
        )
        _style_axis(axis)
        seed = re.search(r"(seed\d+)", curve.run_name.lower())
        seed_label = seed.group(1) if seed else "base run"
        axis.set_title(f"{curve.group_label} ({seed_label})", fontsize=11, pad=7)

    for axis in list(axes.flat)[len(selected) :]:
        axis.axis("off")

    legend = (
        Line2D([0], [0], color="#b7652a", linestyle="--", linewidth=1.5, label="Training loss"),
        Line2D([0], [0], color="#315a91", linewidth=1.5, label="Validation loss"),
    )
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Most stable Run 7 loss curve within each architecture family", fontsize=16, y=0.994)
    fig.supxlabel("Epoch", fontsize=11, y=0.025)
    fig.supylabel("Loss (symmetric log scale)", fontsize=11)
    fig.text(
        0.5,
        0.007,
        "Completed proof-fit runs only; lowest validation-loss MAD over the final 20 epochs per family",
        ha="center",
        fontsize=9,
        color="#465668",
    )
    fig.tight_layout(rect=(0.035, 0.045, 1.0, 0.955), h_pad=1.5, w_pad=1.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("docs/research/run_7_logs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/research/run_7_figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves = load_curves(args.log_root)
    if len(curves) != 111:
        raise RuntimeError(f"Expected 111 Run-7 curves, found {len(curves)}")
    selected = plot_stable_curve_per_family(
        curves,
        args.output_dir / "run_7_stable_loss_curves_by_family.png",
    )
    labels = ", ".join(curve.group_label for curve in selected)
    print(f"Wrote one figure with the most stable completed run per family: {labels}")


if __name__ == "__main__":
    main()
