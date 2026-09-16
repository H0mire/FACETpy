"""Run 6 · Phases D+E — spike preservation, model vs. AAS on identical data.

Evaluates a trained model and an AAS reference on the **validation split** of a
Weg-A dataset and reports the run_6 spike metrics for both.

How AAS is represented
----------------------
AAS removes the epoch-repeatable component of the artifact. In the Weg-A
construction that component is exactly ``artifact_center_template`` — the
AAS+OBS bundle artifact *before* the failure modes were superimposed. The AAS
reference here is therefore ``noisy - template``, which is what an AAS run would
produce **if it recovered the template perfectly**.

That idealisation is deliberate and it favours AAS: a real AAS run also carries
template-estimation noise, and averaging pulls a sliver of every non-periodic
event (including the spikes) into the template. Both effects would make the real
AAS worse than this reference. A model that beats this number therefore beats an
AAS that is strictly better than the real thing — a conservative claim.

Usage::

    uv run python tools/evaluation/eval_run6_spike_preservation.py \
        --dataset output/weg_a_real_v4_512/weg_a_spatiotemporal_dataset.npz \
        --checkpoint training_output/<run>/checkpoints/last.pt \
        --output-dir output/model_evaluations/run6_spike_preservation
"""

from __future__ import annotations

import argparse
import csv
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from facet.training.spike_metrics import compute_spike_metrics, compute_spike_metrics_per_example, expand_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("output/model_evaluations/run6_spike_preservation"))
    p.add_argument(
        "--model-factory",
        default="facet.training.weg_a_baseline:build_model",
        help="module:function returning the model, so any architecture can be evaluated.",
    )
    p.add_argument(
        "--model-kwargs",
        default="{}",
        help="JSON of the architecture kwargs the run was trained with, e.g. '{\"attention_levels\": 2}'. "
        "Defaults differing from the trained config make the checkpoint unloadable.",
    )
    p.add_argument(
        "--max-channels",
        type=int,
        default=None,
        help="Keep only the first N context electrodes, matching data.kwargs.max_channels of the "
        "training config. A model trained on one channel cannot be evaluated on seven.",
    )
    p.add_argument(
        "--residual-mode",
        action="store_true",
        help="Evaluate a cascade model trained with data.kwargs.residual_mode. Such a model sees "
        "the FARM-corrected signal and predicts only what FARM leaves; feeding it the raw signal "
        "instead scores it on data it never saw.",
    )
    p.add_argument(
        "--model-output",
        default=None,
        help="For a model whose forward returns a dict of heads: which key to evaluate as the "
        "predicted artifact. Ignored when the model exposes export_module(), which already "
        "returns the deployed single-tensor view.",
    )
    p.add_argument(
        "--window-shift",
        type=int,
        default=0,
        help="Crop the evaluation window this many samples off the nominal centre (bounded by the "
        "guard band). Everything — input, clean, artifact and template — moves together, so this "
        "asks whether the model depends on the artifact's absolute position in the window. "
        "Training applies a random shift over the whole guard band by default, evaluation does "
        "not, so the nominal number is the best case.",
    )
    p.add_argument(
        "--trigger-misalign",
        type=int,
        default=0,
        help="Shift the signal but NOT the template by this many samples. This is the realistic "
        "failure: a trigger or template estimated a few samples off. Artifact templates are "
        "position-sensitive, so both the FARM arm and any cascade model that consumes "
        "'signal - template' degrade here, while a model reading the raw signal does not.",
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--split",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Which example_split to evaluate: 0 train, 1 validation, 2 locked holdout.",
    )
    p.add_argument("--neighborhood-ms", type=float, default=50.0)
    p.add_argument(
        "--spike-dilate-ms",
        type=float,
        default=0.0,
        help="Widen the spike core label by this many ms in both directions before "
        "computing the metrics. The builder labels only +/-3 samples around the "
        "marker, while the injected IED occupies tens of ms; dilating lets the "
        "morphology metric score the whole waveform instead of the peak core.",
    )
    p.add_argument(
        "--epoch-duration-s",
        type=float,
        default=0.194,
        help="Native trigger-to-trigger epoch length; sets the effective rate (core / duration).",
    )
    return p.parse_args()


def _resolve(spec: str):
    module_name, _, attr = spec.partition(":")
    return getattr(import_module(module_name), attr)


def _load_model(
    checkpoint: Path, input_shape: tuple[int, int, int], device: str, factory: str, kwargs: dict
) -> torch.nn.Module:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    model = _resolve(factory)(input_shape=input_shape, **kwargs)
    model.load_state_dict(state)
    return model.to(device).eval()


class _SingleTensorView(torch.nn.Module):
    """Reduce a multi-head generator to the one tensor the pipeline subtracts.

    A model with several paper outputs (DHCT-GAN has three) cannot be scored
    directly: the metrics need the predicted artifact. Preference order:

    1. ``model.export_module()`` — the model's own statement of what it deploys.
       This is the honest default, because evaluating a head the deployment does
       not use produces a number nobody can reproduce from the exported model.
    2. an explicit ``--model-output`` key, for ablating a specific head.
    """

    def __init__(self, model: torch.nn.Module, output_key: str | None) -> None:
        super().__init__()
        export = getattr(model, "export_module", None)
        self.inner = export() if callable(export) and output_key is None else model
        self.output_key = output_key
        self.via_export = self.inner is not model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        if isinstance(out, dict):
            if self.output_key is None:
                raise SystemExit(
                    f"{type(self.inner).__name__} returned {sorted(out)} and offers no "
                    "export_module(); pass --model-output to say which head is the artifact."
                )
            if self.output_key not in out:
                raise SystemExit(f"--model-output '{self.output_key}' not in {sorted(out)}")
            out = out[self.output_key]
        return out


def main() -> None:
    args = parse_args()
    with np.load(args.dataset, allow_pickle=True) as b:
        if "artifact_center_template" not in b.files:
            raise SystemExit(
                "Dataset lacks 'artifact_center_template'; rebuild with the current builder "
                "so the AAS reference can be derived."
            )
        split = b["example_split"]
        # 0 = train, 1 = validation, 2 = locked holdout. A dataset built with a
        # locked split carries 2 for examples that were kept out of training AND
        # out of early stopping, so evaluating on them is not a re-read of the
        # split the configuration was selected on.
        val = np.flatnonzero(split == args.split)
        core = int(b["core_samples"][0])
        guard = int(b["guard_samples"][0])
        shift = max(-guard, min(int(args.window_shift), guard))
        misalign = max(-guard, min(int(args.trigger_misalign), guard))
        if shift != args.window_shift or misalign != args.trigger_misalign:
            print(f"note: shifts clipped to the guard band of {guard} samples (window {shift}, misalign {misalign})")
        sl = slice(guard + shift, guard + shift + core)
        # The template keeps its own crop so a misalignment can be simulated: the
        # signal moves, the template does not, which is what an imperfect trigger
        # or a drifting template estimate looks like.
        sl_t = slice(guard + shift - misalign, guard + shift - misalign + core)

        def _ctx(key: str) -> np.ndarray:
            """Load one context array for the val split, trimmed to the model's view.

            Each context array is 1.2 GB on disk, so channels and the core window
            are cut at load time: the resident set then holds what the model
            actually sees rather than three full copies of the dataset.
            """
            arr = b[key][val]  # (n, ep, ch, L)
            if args.max_channels is not None:
                # Must mirror the training config: the builder writes the target
                # electrode first, so a prefix is "target plus nearest N-1 neighbours".
                limit = max(1, min(int(args.max_channels), arr.shape[2]))
                arr = arr[:, :, :limit]
            window = sl_t if key.endswith("_template") else sl
            return arr[..., window]

        clean_ctx = _ctx("clean_context")  # (n, ep, ch, core)
        artifact_ctx = _ctx("artifact_context")
        if args.residual_mode:
            if "artifact_context_template" not in b.files:
                raise SystemExit(
                    "--residual-mode needs 'artifact_context_template': the cascade input is "
                    "clean + (artifact - template) and cannot be reconstructed without it."
                )
            # Exactly what NPZSpatioTemporalDataset(residual_mode=True) builds.
            artifact_ctx = artifact_ctx - _ctx("artifact_context_template")
        clean = b["clean_center"][val][:, 0, sl]  # true clean (with IEDs)
        artifact = b["artifact_center"][val][:, 0, sl]  # enriched artifact
        template = b["artifact_center_template"][val][:, 0, sl_t]  # AAS-removable part
        spikes = b["spike_labels"][val][:, 0, sl]
        if args.spike_dilate_ms > 0:
            # Scored extent of a spike, widened. The label marks the marker
            # sample +/-3; a real IED lasts 20-200 ms. Widening the *label* is
            # the honest way to score the whole event, because it changes what
            # is measured without touching the signal or re-injecting anything.
            dilate = int(round(args.spike_dilate_ms * 1e-3 * core / args.epoch_duration_s))
            spikes = expand_mask(spikes > 0, max(1, dilate)).astype(spikes.dtype)
        # Clustering keys. The spike-bearing examples are not independent: the
        # builder emits one example per target electrode, so the same injected
        # event appears once per electrode. A test that treats those as
        # independent pairs is anti-conservative by roughly the number of
        # electrodes, so the event id has to travel with every per-example row.
        center_epoch = b["center_epoch_index"][val]
        target_channel = b["target_channel_index"][val]

    noisy_ctx = clean_ctx + artifact_ctx
    noisy = clean + artifact

    # --- model ---
    model = _load_model(
        args.checkpoint,
        (noisy_ctx.shape[1], noisy_ctx.shape[2], core),
        args.device,
        args.model_factory,
        json.loads(args.model_kwargs),
    )
    model = _SingleTensorView(model, args.model_output).to(args.device).eval()
    preds = np.empty_like(artifact)
    with torch.no_grad():
        for i in range(0, noisy_ctx.shape[0], args.batch_size):
            batch = torch.from_numpy(noisy_ctx[i : i + args.batch_size]).to(args.device)
            preds[i : i + args.batch_size] = model(batch).cpu().numpy()[:, 0, :]
    corrected_aas = noisy - template
    # In residual mode the model's output is subtracted from the FARM-corrected
    # signal, not from the raw one, so the model arm is literally "what the
    # cascade adds on top of its own input" and the AAS arm is that input.
    corrected_model = (corrected_aas if args.residual_mode else noisy) - preds

    # Each epoch is resampled to `core` samples, so the effective rate is
    # core / epoch_duration, not the bundle rate.
    margin = int(round(args.neighborhood_ms * 1e-3 * core / args.epoch_duration_s))
    margin = max(1, min(margin, core // 3))

    # Third arm: the trivial "return nothing" corrector. On this data the artifact
    # is ~56x the clean, so outputting zero clean scores exactly RMS(clean) and is
    # a strong opponent that smoothness-rewarding metrics do not see. Two runs have
    # already beaten FARM on the headline metrics while losing to this baseline.
    corrected_null = np.zeros_like(clean)

    arms = {"model": corrected_model, "aas_ideal": corrected_aas, "null_output": corrected_null}
    results = {
        name: compute_spike_metrics(est, clean, spikes, neighborhood_samples=margin) for name, est in arms.items()
    }
    per_example = {
        name: compute_spike_metrics_per_example(est, clean, spikes, neighborhood_samples=margin)
        for name, est in arms.items()
    }
    for name, res in results.items():
        res["overall_rmse_uv"] = float(np.sqrt(np.mean((arms[name] - clean).astype(np.float64) ** 2))) * 1e6

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Per-example values as CSV, one row per spike-bearing example per arm. This is
    # what makes a paired comparison possible at all: the aggregate JSON cannot
    # support a paired test, an effect size or an interval.
    csv_path = args.output_dir / "run6_spike_preservation_per_example.csv"
    keys = [k for k in per_example["model"] if k != "example_index"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arm", "example_index", "spike_event_id", "target_channel", *keys])
        for name, block in per_example.items():
            for row in range(block["example_index"].size):
                i = int(block["example_index"][row])
                writer.writerow(
                    [name, i, int(center_epoch[i]), int(target_channel[i]), *(f"{block[k][row]:.10g}" for k in keys)]
                )
    # Bulk per-example table, covering ALL validation examples rather than only
    # the spike-bearing ones. The headline artifact-correction claim lives here:
    # the validation split holds 162 epoch-disjoint centre epochs, so a paired
    # test at epoch level is properly powered — unlike the spike metrics, whose
    # independent unit count is the number of injected events.
    bulk_path = args.output_dir / "run6_bulk_per_example.csv"
    with bulk_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arm", "example_index", "epoch_id", "target_channel", "rmse_uv", "clean_snr_db", "has_spike"])
        clean_power = np.mean(clean.astype(np.float64) ** 2, axis=1)
        spike_any = spikes.any(axis=1)
        for name, est in arms.items():
            err = (est - clean).astype(np.float64)
            rmse = np.sqrt(np.mean(err**2, axis=1)) * 1e6
            err_power = np.mean(err**2, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = 10.0 * np.log10(np.where(err_power > 0, clean_power / err_power, np.inf))
            for i in range(clean.shape[0]):
                writer.writerow(
                    [
                        name,
                        i,
                        int(center_epoch[i]),
                        int(target_channel[i]),
                        f"{rmse[i]:.10g}",
                        f"{snr[i]:.10g}",
                        int(spike_any[i]),
                    ]
                )

    spike_rows = per_example["model"]["example_index"].astype(int)
    events = sorted({int(center_epoch[i]) for i in spike_rows})
    meta_per_example = {
        "per_example_csv": csv_path.name,
        "n_rows_per_arm": int(spike_rows.size),
        "n_spike_events": len(events),
        "spike_event_ids": events,
        "rows_per_event": {str(e): int(sum(1 for i in spike_rows if int(center_epoch[i]) == e)) for e in events},
        "clustering_note": "One row per target electrode per injected event. The event id is the "
        "unit of independence; rows sharing it are replicates of one spike.",
        "bulk_per_example_csv": bulk_path.name,
        "n_bulk_rows_per_arm": int(clean.shape[0]),
        "n_bulk_epochs": len({int(e) for e in center_epoch}),
    }

    meta = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "model_factory": args.model_factory,
        "model_kwargs": json.loads(args.model_kwargs),
        "max_channels": args.max_channels,
        "window_shift_samples": shift,
        "trigger_misalign_samples": misalign,
        "shift_note": "window_shift moves signal and template together; trigger_misalign moves "
        "only the signal, simulating a template estimated off-position",
        "residual_mode": bool(args.residual_mode),
        "model_arm": (
            "noisy - template - model(clean + artifact - template)   [cascade]"
            if args.residual_mode
            else "noisy - model(clean + artifact)"
        ),
        "model_output": args.model_output if not model.via_export else "export_module()",
        "n_val_examples": int(val.size),
        "neighborhood_samples": margin,
        "spike_dilate_ms": args.spike_dilate_ms,
        "spike_dilate_samples": int(round(args.spike_dilate_ms * 1e-3 * core / args.epoch_duration_s))
        if args.spike_dilate_ms > 0
        else 0,
        "aas_reference": "noisy - artifact_center_template (ideal AAS: perfect template recovery)",
        "null_reference": "clean_hat = 0 (trivial corrector; overall_rmse_uv equals RMS(clean))",
        "per_example": meta_per_example,
        "results": results,
    }
    out_path = args.output_dir / "run6_spike_preservation.json"
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    keys = [
        ("spike_neighborhood_snr_db", "spike neighbourhood SNR (dB)  [HEADLINE]"),
        ("spike_contrast_db", "spike-to-residual contrast (dB) [HEADLINE]"),
        ("non_spike_snr_db", "non-spike SNR (dB)"),
        ("spike_peak_over_residual", "spike peak / local residual (>1 = usable)"),
        ("spike_amplitude_ratio", "spike amplitude ratio (1.0 = intact)"),
        ("spike_morphology_corr", "spike morphology corr"),
        ("spike_peak_latency_drift_samples", "peak latency drift (samples)"),
        ("overall_rmse_uv", "overall RMSE (uV)"),
    ]
    print(
        f"\nRun 6 · spike preservation — {int(val.size)} validation examples, "
        f"{int(results['model']['n_spike_examples'])} with spikes\n"
    )
    print(f"{'metric':<42}{'model':>12}{'AAS (ideal)':>14}")
    print("-" * 68)
    for key, label in keys:
        print(f"{label:<42}{results['model'][key]:>12.3f}{results['aas_ideal'][key]:>14.3f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
