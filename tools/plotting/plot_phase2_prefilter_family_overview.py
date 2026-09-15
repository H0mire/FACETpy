#!/usr/bin/env python3
"""Render Run-7 deployment families before the final 70-Hz low-pass."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from facet.core import Pipeline  # noqa: E402
from facet.correction import DeepLearningCorrection  # noqa: E402
from pipeline_demo import reference_chain  # noqa: E402
from pipeline_demo.family_adapters import DEPLOYMENT_SPECS, FamilyAdapter  # noqa: E402


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


def context_for(model_id: str):
    # CPU avoids MPS float64 conversion failures in the reconstructed EDF path.
    device = "cpu"
    correction = DeepLearningCorrection(model=FamilyAdapter(model_id, device=device))
    # Keep the normal PCA cleanup, but stop immediately before the final 70-Hz
    # low-pass.  This is the actual pipeline output at the requested state.
    steps = (reference_chain.preprocessing(EDF) + [correction]
             + reference_chain.postprocessing(include_pca=True, include_lowpass=False))
    result = Pipeline(steps, name=f"{model_id}_pre_lowpass").run()
    if not result.success:
        raise RuntimeError(result.error)
    return result.context


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Input state sent to all direct deployment models: no correction and no final LP.
    ref_result = Pipeline(reference_chain.preprocessing(EDF)
                          + reference_chain.postprocessing(include_pca=False, include_lowpass=False),
                          name="phase2_input_pre_lowpass").run()
    if not ref_result.success:
        raise RuntimeError(ref_result.error)
    ref = ref_result.context
    raw = ref.get_raw()
    sfreq = float(raw.info["sfreq"])
    names = list(raw.ch_names)
    channel = names.index("Fp1") if "Fp1" in names else 0
    triggers = np.asarray(ref.metadata.triggers, dtype=float)
    trigger = float(triggers[np.argmin(np.abs(triggers / sfreq - 31.5))])
    start = int(round(trigger + ref.metadata.artifact_to_trigger_offset * sfreq))
    artifact_length = int(ref.metadata.artifact_length)
    stop = min(raw.n_times, start + artifact_length)
    time = (np.arange(start, stop) - trigger) / sfreq * 1000
    input_seg = raw.get_data(picks=[channel], start=start, stop=stop)[0] * 1e6

    outputs: dict[str, np.ndarray] = {}
    for model_id, label in MODELS:
        print(f"[run] {label}", flush=True)
        ctx = context_for(model_id)
        mne_raw = ctx.get_raw()
        outputs[model_id] = mne_raw.get_data(picks=[channel], start=start, stop=stop)[0] * 1e6

    lim = np.quantile(np.abs(np.concatenate([input_seg, *outputs.values()])), .996) * 1.08
    fig, axes = plt.subplots(4, 4, figsize=(13.2, 9.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (model_id, label) in zip(axes, MODELS):
        ax.plot(time, input_seg, color="#a9a9a9", lw=.55, label="Model input (pre-low-pass)")
        ax.plot(time, outputs[model_id], color="#2d6fae", lw=.85, label="Pipeline output (pre-low-pass)")
        ax.set_title(label, fontsize=8.8, fontweight="bold")
        ax.grid(alpha=.18, lw=.45)
        ax.set_ylim(-lim, lim)
    for ax in axes[len(MODELS):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(.5,.015))
    fig.suptitle("Phase 2: deployment-family outputs before final low-pass", fontsize=14,
                 fontweight="bold", y=.995)
    fig.text(.5,.065,"Time relative to trigger (ms)",ha="center",fontsize=10)
    fig.text(.012,.5,"Amplitude (µV)",va="center",rotation="vertical",fontsize=10)
    fig.text(.5,.085,"One documented artifact-length window; 1-Hz high-pass and trigger alignment retained; no final 70-Hz low-pass",
             ha="center",fontsize=8.5)
    fig.tight_layout(rect=(.025,.10,1,.96))
    path = OUT / "figure_phase2_families_prefilter_artifact_window.png"
    fig.savefig(path,dpi=220,bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
