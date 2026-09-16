#!/usr/bin/env python3
"""Fast per-family Phase-2 view for one real artifact epoch before postprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from masterthesis_guide.reproduce import adapter as catalog_adapter  # noqa: E402

from facet.core import Pipeline  # noqa: E402
from facet.correction.deep_learning import _resample_1d  # noqa: E402
from facet.models.masterthesis import pipeline as reference_chain  # noqa: E402
from facet.models.masterthesis.adapters import (  # noqa: E402 - repository path is set before checkout-only imports
    FamilyAdapter,
    predict_from_context,
)

EDF = ROOT / "examples/datasets/NiazyFMRI.edf"
OUT = ROOT / "output/thesis_results_by_phase/phase_2_pipeline_deployment"
MODELS = [
    ("st_gnn_deployment", "ST-GNN"),
    ("nested_gan_deployment", "Nested GAN"),
    ("vit_spectrogram_deployment", "Vision Transformer"),
    ("sepformer_deployment", "SepFormer"),
    ("dhct_gan_deployment", "DHCT-GAN"),
    ("dhct_gan_v2_deployment", "DHCT-GAN v2"),
    ("cascaded_context_dae_deployment", "Cascaded Context DAE"),
    ("demucs_deployment", "Demucs"),
    ("conv_tasnet_deployment", "Conv-TasNet"),
    ("denoise_mamba_deployment", "DenoiseMamba"),
    ("dpae_deployment", "DPAE"),
    ("cascaded_dae_deployment", "Cascaded DAE"),
    ("ic_unet_deployment", "IC-U-Net"),
]


def one_prediction(adapter: FamilyAdapter, context, centre: int) -> tuple[np.ndarray, int, int]:
    raw = context.get_raw()
    starts, stops, target_samples = adapter._build_epoch_boundaries(
        context, np.asarray(context.get_triggers(), dtype=int), raw.n_times
    )
    ids = adapter._context_indices(centre, len(starts), adapter.context_epochs // 2)
    channels = adapter._resolve_channels(raw)
    data = raw._data
    packed = np.stack(
        [
            np.stack([_resample_1d(data[ch, starts[e] : stops[e]], target_samples) for ch in channels], axis=0)
            for e in ids
        ],
        axis=0,
    )[None]
    model, _ = adapter._load_model()
    predicted = predict_from_context(
        adapter.packing, model, packed, device=adapter.device, batch_size=adapter.batch_size
    )[0]
    length = stops[centre] - starts[centre]
    prediction = np.stack([_resample_1d(predicted[k], length) for k in range(len(channels))])
    return prediction, int(starts[centre]), int(stops[centre])


def main() -> None:
    import argparse

    global EDF, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    EDF, OUT = args.edf, args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    result = Pipeline(reference_chain.preprocessing(EDF), name="phase2_prefilter_input").run()
    if not result.success:
        raise RuntimeError(result.error)
    context = result.context
    raw = context.get_raw()
    sfreq = float(raw.info["sfreq"])
    names = list(raw.ch_names)
    channel = names.index("Fp1") if "Fp1" in names else 0
    base_adapter = catalog_adapter("deployment_dpae", device="cpu")
    starts, stops, _ = base_adapter._build_epoch_boundaries(
        context, np.asarray(context.get_triggers(), dtype=int), raw.n_times
    )
    target = float(
        np.asarray(context.get_triggers())[np.argmin(np.abs(np.asarray(context.get_triggers()) / sfreq - 31.5))]
    )
    centre = int(np.argmin(np.abs(starts - (target + context.metadata.artifact_to_trigger_offset * sfreq))))
    start, stop = int(starts[centre]), int(stops[centre])
    time = (np.arange(start, stop) - target) / sfreq * 1000
    input_seg = raw._data[channel, start:stop] * 1e6
    output: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for model_id, label in MODELS:
        print(f"[run] {label}", flush=True)
        try:
            # Single-window inference is small; CPU also avoids MPS float64
            # limitations embedded in several TorchScript exports.
            eid = "deployment_" + model_id.removesuffix("_deployment")
            if model_id == "dhct_gan_deployment":
                eid = "run8_dhct_gan_lr0_0001_bc8_sisdr0_s42"
            adapter = catalog_adapter(eid, device="cpu")
            prediction, pred_start, pred_stop = one_prediction(adapter, context, centre)
            if (pred_start, pred_stop) != (start, stop):
                raise RuntimeError("selected epoch boundaries diverged")
            output[model_id] = input_seg - prediction[channel] * 1e6
        except Exception as exc:  # keep a visual record of individual unavailable arms
            failures[model_id] = str(exc).splitlines()[-1]
            print(f"[fail] {label}: {failures[model_id]}", flush=True)
    # The raw artifact contains brief, high-amplitude edge excursions.  A
    # quantile limit clips precisely the structure this diagnostic needs to
    # show, so retain the full shared physical range instead.
    # The input defines the physical reference scale.  Model outputs are shown
    # relative to that same raw-artifact range, never allowed to set it.
    lim = np.max(np.abs(input_seg)) * 1.08

    def render(path: Path, *, lowpass_outputs: bool) -> None:
        fig, axes = plt.subplots(4, 4, figsize=(13.2, 9.4), sharex=True, sharey=True)
        sos = butter(4, 70.0, btype="lowpass", fs=sfreq, output="sos")
        for ax, (model_id, label) in zip(axes.ravel(), MODELS, strict=False):
            corrected = output[model_id]
            if lowpass_outputs:
                corrected = sosfiltfilt(sos, corrected)
            ax.plot(time, input_seg, color="#a9a9a9", lw=0.55, label="Model input (unfiltered)")
            ax.plot(
                time,
                corrected,
                color="#2d6fae",
                lw=0.85,
                label="Model-corrected output (70-Hz low-pass)" if lowpass_outputs else "Model-corrected output",
            )
            ax.set_title(label, fontsize=8.8, fontweight="bold")
            ax.set_ylim(-lim, lim)
            ax.grid(alpha=0.18, lw=0.45)
        for ax in axes.ravel()[len(MODELS) :]:
            ax.axis("off")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.015))
        title = (
            "Phase 2: model corrections with 70-Hz output low-pass"
            if lowpass_outputs
            else "Phase 2: model corrections on one artifact-length window"
        )
        note = (
            "Unfiltered input; only the corrected model output is 70-Hz low-pass filtered"
            if lowpass_outputs
            else "Actual model input before PCA cleanup and final 70-Hz low-pass; 1-Hz high-pass and trigger alignment retained"
        )
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
        fig.text(0.5, 0.065, "Time relative to trigger (ms)", ha="center", fontsize=10)
        fig.text(0.012, 0.5, "Amplitude (µV)", va="center", rotation="vertical", fontsize=10)
        fig.text(0.5, 0.085, note, ha="center", fontsize=8.5)
        fig.tight_layout(rect=(0.025, 0.10, 1, 0.96))
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    render(OUT / "figure_phase2_families_artifact_window_prefilter.png", lowpass_outputs=False)
    filtered_path = OUT / "figure_phase2_families_artifact_window_70hz_output_lowpass.png"
    render(filtered_path, lowpass_outputs=True)
    print(filtered_path)
    if failures:
        raise RuntimeError(f"Unavailable model outputs: {failures}")


if __name__ == "__main__":
    main()
