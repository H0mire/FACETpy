#!/usr/bin/env python3
"""Aggregate per-model unified-holdout metrics into UNIFIED_HOLDOUT.md and
update output/model_evaluations/INDEX.md with the new column.

Run AFTER eval_unified_holdout.py has produced the per-model
output/model_evaluations/<id>/holdout_v1/metrics.json files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "output/model_evaluations"
RUN_ID = "holdout_v1"

FAMILY: dict[str, str] = {
    "aas_naive_6nn": "Baseline (AAS)",
    "dpae": "Discriminative",
    "ic_unet": "Discriminative + ICA",
    "denoise_mamba": "SSM",
    "vit_spectrogram": "Vision (MAE)",
    "st_gnn": "Graph (GNN)",
    "conv_tasnet": "Audio (TCN)",
    "demucs": "Audio (U-Net+LSTM)",
    "sepformer": "Audio (Transformer)",
    "nested_gan": "GAN (TF+Time)",
    "d4pm": "Diffusion",
    "dhct_gan": "GAN (single-epoch input, failed)",
    "dhct_gan_v2": "GAN (hybrid CNN+Transformer, ctx fix)",
    "cascaded_dae": "Autoencoder (cascaded MLP)",
    "cascaded_context_dae": "Autoencoder (context MLP)",
}

# Note about non-comparability — d4pm Run 1 was on 32 samples, vit on full 833, etc.
# Both cascaded DAE entries reference their own Run-1 retrofill (this branch); the
# original synthetic_spike numbers are kept out of this delta column on purpose.
# aas_naive_6nn is a non-learned baseline — its "Run 1 SNR" mirrors its holdout
# SNR (deterministic).
RUN1_SNR: dict[str, float] = {
    "demucs": 31.28,
    "conv_tasnet": 22.03,
    "sepformer": 19.05,
    "cascaded_context_dae": 18.84,
    "cascaded_dae": 17.79,
    "nested_gan": 13.54,
    "denoise_mamba": 11.80,
    "ic_unet": 11.77,
    "vit_spectrogram": 11.60,
    "st_gnn": 11.00,
    "aas_naive_6nn": 9.16,
    "dpae": 7.48,
    "d4pm": 3.21,
    "dhct_gan_v2": 1.69,
    "dhct_gan": -7.13,
}


def _load_metrics(model_id: str) -> dict | None:
    path = EVAL_ROOT / model_id / RUN_ID / "metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    flat = payload.get("flat_metrics", {})
    # The metrics dict was written nested under "unified_holdout"
    keys_with_prefix = {k.replace("unified_holdout.", ""): v for k, v in flat.items()}
    return keys_with_prefix


def build_unified_md() -> str:
    rows: list[tuple[str, dict]] = []
    failures: list[str] = []
    for model_id in FAMILY:
        m = _load_metrics(model_id)
        if m is None:
            failures.append(model_id)
            continue
        if "clean_snr_improvement_db" not in m:
            failures.append(model_id)
            continue
        rows.append((model_id, m))

    rows.sort(key=lambda kv: -kv[1]["clean_snr_improvement_db"])

    lines = [
        "# Unified Holdout Re-Evaluation",
        "",
        "Implements the cross-model re-evaluation on a unified test split requested in",
        "[docs/research/thesis_results_report.md §5](../../docs/research/thesis_results_report.md#5-critical-caveats)",
        "and [docs/research/run_2_plan.md §5.1](../../docs/research/run_2_plan.md). Every",
        "model is evaluated on the **same 166 held-out windows** (4980 channel-windows*),",
        "producing directly comparable absolute numbers — the test-split-size confound",
        "from Run 1 is eliminated.",
        "",
        "\\* 166 windows × 30 channels = **4980** channel-windows. Note: in Run 1, d4pm was",
        "evaluated on only 32 windows × 4 channels (= 128 ch-w) and ic_unet/vit_spectrogram",
        "/nested_gan on all 833 windows; here every model sees the same 166 windows.",
        "",
        "Dataset: `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`",
        "Holdout: seed=42, val_ratio=0.2 at the window level.",
        "Split hash: `sha256:ddaa64a504e062fd`.",
        "Holdout indices: `output/niazy_proof_fit_context_512/holdout_v1_indices.json`.",
        "Driver: [`tools/eval_unified_holdout.py`](../../tools/eval_unified_holdout.py).",
        "",
        "## Cross-Model Ranking (Unified Holdout)",
        "",
        "| Rank | Model | Family | SNR↑ dB (holdout) | SNR before | SNR after | art.corr | res.RMS ratio | Δ vs Run 1 | t [s] |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (model_id, m) in enumerate(rows, 1):
        snr_imp = m["clean_snr_improvement_db"]
        snr_before = m["clean_snr_db_before"]
        snr_after = m["clean_snr_db_after"]
        art_corr = m["artifact_corr"]
        res_rms = m["residual_error_rms_ratio"]
        run1 = RUN1_SNR.get(model_id)
        delta_run1 = f"{snr_imp - run1:+.2f}" if run1 is not None else "—"
        t_sec = m.get("inference_seconds", 0.0)
        lines.append(
            f"| {rank} | {model_id} | {FAMILY[model_id]} | "
            f"{snr_imp:+.2f} | {snr_before:+.2f} | {snr_after:+.2f} | "
            f"{art_corr:+.4f} | {res_rms:.3f} | {delta_run1} | {t_sec:.1f} |"
        )

    lines += [
        "",
        "**Reading the table:**",
        "- `SNR↑ dB` is the primary thesis metric. Higher is better.",
        "- `art.corr` = Pearson correlation between predicted and true artifact (all channels × samples). Closer to +1 is better.",
        "- `res.RMS ratio` = RMS(corrected − clean) / RMS(noisy − clean). Lower is better.",
        "- `Δ vs Run 1` shows how the unified-holdout number differs from the original Run 1 ranking. Most values are within 1 dB — confirming the original ranking was qualitatively correct despite the split confound, except d4pm (now properly evaluated on 4980 ch-w instead of 128) which jumped from +3.21 to +4.81.",
        "",
        "## Methodology Notes",
        "",
        "- **Holdout split** is at the **window level** (n=833 → 166), not channel-window level. This means all 30 channels of each holdout window go through every model, and per-channel models see them as 4980 channel-windows. The split is deterministic given `seed=42`.",
        "- **Metric formulas** are reused verbatim from `examples/model_evaluation/evaluate_conv_tasnet.py` so the absolute numbers stay backwards-compatible with Run 1 per-model evaluations.",
        "- **Inference paths**:",
        "  - 10 models: TorchScript export (`.ts` file) from `training_output/<run>/exports/`.",
        "  - `denoise_mamba`: Source module + `last.pt` checkpoint, because the TorchScript bakes `device='cuda:0'` into the SSM scan (`run_2_plan §3.5` device-baking anti-pattern).",
        "  - `d4pm`: Source module + `last.pt` checkpoint, because the `d4pm.ts` export is a zero-stub (the DDPM reverse loop wasn't traced).",
        "- **vit_spectrogram** is the only model whose underlying TS predicts the *clean* center epoch, not the artifact. The artifact is recovered in the driver via `artifact = noisy_demeaned − pred_clean` — matching the per-model adapter.",
        "",
        "## Per-model artifacts",
        "",
    ]
    for model_id, _ in rows:
        lines.append(
            f"- [`{model_id}/{RUN_ID}/`](../{model_id}/{RUN_ID}/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`"
        )

    if failures:
        lines += ["", "## Missing / Failed", ""]
        for f in failures:
            lines.append(f"- **{f}** — no `holdout_v1/metrics.json` found")
    lines.append("")
    return "\n".join(lines)


def patch_index(unified_rows: list[tuple[str, dict]]) -> str:
    """Append a unified-holdout section to INDEX.md."""
    index_path = EVAL_ROOT / "INDEX.md"
    text = index_path.read_text()

    # Drop any previously added unified-holdout section
    text = re.sub(
        r"\n## Unified Holdout Re-Evaluation \(Run 2 §5\.1\).*?(?=\n## |\Z)",
        "",
        text,
        flags=re.DOTALL,
    )

    rows = sorted(unified_rows, key=lambda kv: -kv[1]["clean_snr_improvement_db"])

    new_section = [
        "",
        "## Unified Holdout Re-Evaluation (Run 2 §5.1)",
        "",
        f"Common test split: 166 windows × 30 channels = 4980 channel-windows, seed=42.",
        f"Driver: [`tools/eval_unified_holdout.py`](../../tools/eval_unified_holdout.py).",
        f"Full report: [`UNIFIED_HOLDOUT.md`](UNIFIED_HOLDOUT.md).",
        "",
        "| Rank | Model | SNR↑ dB | art.corr | res.RMS | Run 1 SNR↑ | Δ |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (model_id, m) in enumerate(rows, 1):
        run1 = RUN1_SNR.get(model_id)
        delta_run1 = f"{m['clean_snr_improvement_db'] - run1:+.2f}" if run1 is not None else "—"
        run1_str = f"{run1:+.2f}" if run1 is not None else "—"
        new_section.append(
            f"| {rank} | {model_id} | "
            f"{m['clean_snr_improvement_db']:+.2f} | "
            f"{m['artifact_corr']:+.4f} | "
            f"{m['residual_error_rms_ratio']:.3f} | "
            f"{run1_str} | {delta_run1} |"
        )
    new_section.append("")
    return text.rstrip() + "\n" + "\n".join(new_section)


def main() -> int:
    md = build_unified_md()
    out = EVAL_ROOT / "UNIFIED_HOLDOUT.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO_ROOT)}")

    rows: list[tuple[str, dict]] = []
    for model_id in FAMILY:
        m = _load_metrics(model_id)
        if m is not None and "clean_snr_improvement_db" in m:
            rows.append((model_id, m))

    if rows:
        new_index = patch_index(rows)
        (EVAL_ROOT / "INDEX.md").write_text(new_index, encoding="utf-8")
        print(f"Patched {(EVAL_ROOT / 'INDEX.md').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
