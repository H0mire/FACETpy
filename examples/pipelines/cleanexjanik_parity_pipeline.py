"""
CleanExJanik parity example (MATLAB FACET vs FACETpy).

This script builds a FACETpy pipeline in the style of CleanExJanik.m,
runs it with ``channel_sequential=True`` and classic logging, and compares
intermediate outputs against MATLAB reference EDFs in ``examples/datasets``.

Outputs:
  - ./output/cleanexjanik_parity/corrected_cleanexjanik_equivalent.edf
  - ./output/cleanexjanik_parity/comparison.json
  - ./output/cleanexjanik_parity/findings.md
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.signal import filtfilt

# Configure classic logging before importing facet.
os.environ.setdefault("FACET_CONSOLE_MODE", "classic")
os.environ.setdefault("FACET_LOG_FILE", "1")
os.environ.setdefault("FACET_LOG_CONSOLE_LEVEL", "INFO")
os.environ.setdefault("FACET_LOG_DIR", "./output/cleanexjanik_parity/logs")

from facet import (  # noqa: E402
    AASCorrection,
    ANCCorrection,
    CutAcquisitionWindow,
    DownSample,
    EDFExporter,
    HighPassFilter,
    Loader,
    LowPassFilter,
    PCACorrection,
    PasteAcquisitionWindow,
    Pipeline,
    RawTransform,
    SliceAligner,
    SubsampleAligner,
    TriggerDetector,
    UpSample,
)

INPUT_FILE = "./examples/datasets/NiazyFMRI.set"
OUTPUT_DIR = Path("./output/cleanexjanik_parity")
OUTPUT_FILE = str(OUTPUT_DIR / "corrected_cleanexjanik_equivalent.edf")
COMPARISON_JSON = OUTPUT_DIR / "comparison.json"
REPORT_FILE = OUTPUT_DIR / "findings.md"

UPSAMPLE = 10
TRIGGER_REGEX = r"\b1\b"

REFERENCE_FILES: dict[str, str] = {
    "alignment": "./examples/datasets/matlab_with_alignment.edf",
    "aas": "./examples/datasets/matlab_only_aas.edf",
    "pca": "./examples/datasets/matlab_with_pca.edf",
    "lowpass": "./examples/datasets/matlab_only_lowpass.edf",
    "anc": "./examples/datasets/matlab_with_anc.edf",
}


@dataclass
class StageComparison:
    stage: str
    channels_compared: int
    samples_compared: int
    sfreq_hz: float
    artifact_length_samples: int | None
    mae: float
    rmse: float
    rel_rmse_percent: float
    max_abs: float
    median_corr: float | None
    p05_corr: float | None
    p95_corr: float | None


class ParityProbe:
    """Collect step-wise comparisons during a single pipeline run."""

    def __init__(self) -> None:
        self._ref_cache: dict[str, mne.io.BaseRaw] = {}
        self.stages: dict[str, StageComparison] = {}

    def _load_reference(self, stage: str) -> mne.io.BaseRaw:
        path = REFERENCE_FILES[stage]
        if path not in self._ref_cache:
            self._ref_cache[path] = mne.io.read_raw_edf(path, preload=True, verbose=False)
        return self._ref_cache[path]

    def checkpoint(self, stage: str, to_native: bool = False):
        """Return a callable pipeline step that logs comparison for one stage."""

        def _step(context):
            self.compare(stage=stage, context=context, to_native=to_native)
            return context

        _step.__name__ = f"checkpoint_{stage}"
        return _step

    def compare(self, stage: str, context, to_native: bool = False) -> None:
        """Compare current context against one MATLAB reference EDF."""
        compare_ctx = context
        if to_native:
            compare_ctx = DownSample(factor=UPSAMPLE).execute(compare_ctx)
            compare_ctx = PasteAcquisitionWindow().execute(compare_ctx)

        raw = compare_ctx.get_raw()
        ref = self._load_reference(stage)

        channel_names = [ch for ch in raw.ch_names if ch in ref.ch_names and ch != "Status"]
        raw_picks = [raw.ch_names.index(ch) for ch in channel_names]
        ref_picks = [ref.ch_names.index(ch) for ch in channel_names]

        data_raw = raw.get_data(picks=raw_picks)
        data_ref = ref.get_data(picks=ref_picks)
        n_samples = min(data_raw.shape[1], data_ref.shape[1])
        data_raw = data_raw[:, :n_samples]
        data_ref = data_ref[:, :n_samples]

        diff = data_raw - data_ref
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        max_abs = float(np.max(np.abs(diff)))

        ref_rms = float(np.sqrt(np.mean(data_ref * data_ref)))
        rel_rmse = (rmse / ref_rms * 100.0) if ref_rms > 0 else float("nan")

        corrs: list[float] = []
        for i in range(diff.shape[0]):
            a = data_raw[i]
            b = data_ref[i]
            if np.std(a) <= 0 or np.std(b) <= 0:
                continue
            c = np.corrcoef(a, b)[0, 1]
            if np.isfinite(c):
                corrs.append(float(c))

        self.stages[stage] = StageComparison(
            stage=stage,
            channels_compared=len(channel_names),
            samples_compared=int(n_samples),
            sfreq_hz=float(raw.info["sfreq"]),
            artifact_length_samples=int(context.metadata.artifact_length)
            if context.metadata.artifact_length
            else None,
            mae=mae,
            rmse=rmse,
            rel_rmse_percent=float(rel_rmse),
            max_abs=max_abs,
            median_corr=float(np.median(corrs)) if corrs else None,
            p05_corr=float(np.percentile(corrs, 5)) if corrs else None,
            p95_corr=float(np.percentile(corrs, 95)) if corrs else None,
        )


def _mark_status_as_stim(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Map EEGLAB 'Status' channel to STIM so TriggerDetector can read it."""
    out = raw.copy()
    if "Status" in out.ch_names:
        out.set_channel_types({"Status": "stim"})
    return out


def compute_anc_mu_diagnostics(context) -> dict[str, Any]:
    """Recompute ANC learning-rate diagnostics for FACETpy and MATLAB-equivalent N."""
    raw = context.get_raw()
    noise = context.get_estimated_noise()
    if noise is None:
        return {"error": "No estimated noise available; ANC diagnostics skipped."}

    anc_meta = context.metadata.custom.get("anc", {})
    anc_probe = ANCCorrection(use_c_extension=True)

    artifact_length = int(context.get_artifact_length() or 1)
    sfreq = float(context.get_sfreq())
    hp_weights, hp_cutoff = anc_probe._resolve_hp_filter(context, artifact_length, sfreq)
    s_acq_start, s_acq_end = anc_probe._get_acquisition_window(context)

    n_current = int(anc_meta.get("filter_order", max(1, artifact_length)))
    # MATLAB ANCFilterOrder is in native sample units at ANC stage.
    n_matlab_equivalent = max(1, int(artifact_length))
    # Diagnostic reference for legacy behavior where artifact_length still
    # carried upsampled units at ANC stage.
    n_legacy_upsampled_unit = max(1, int(np.ceil(artifact_length / UPSAMPLE)))

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    mu_current: list[float] = []
    mu_matlab: list[float] = []

    for ch in picks:
        eeg = raw._data[ch]
        reference = noise[ch, s_acq_start:s_acq_end].astype(float, copy=True)
        if reference.size == 0:
            continue

        if hp_weights is not None:
            data = filtfilt(hp_weights, 1, eeg, axis=0, padtype="odd")[s_acq_start:s_acq_end].astype(float)
        else:
            data = eeg[s_acq_start:s_acq_end].astype(float)

        ref_energy = np.sum(reference * reference)
        if not np.isfinite(ref_energy) or ref_energy <= np.finfo(float).eps:
            continue

        alpha = np.sum(data * reference) / ref_energy
        if not np.isfinite(alpha):
            continue

        reference_scaled = alpha * reference
        var_ref = np.var(reference_scaled)
        if not np.isfinite(var_ref) or var_ref <= np.finfo(float).eps:
            continue

        mu_current.append(float(0.05 / (n_current * var_ref)))
        mu_matlab.append(float(0.05 / (n_matlab_equivalent * var_ref)))

    if not mu_current:
        return {
            "hp_cutoff_hz": float(hp_cutoff),
            "n_current_facetpy": n_current,
            "n_matlab_equivalent": n_matlab_equivalent,
            "error": "Could not derive finite mu values.",
        }

    ratio = np.array(mu_matlab) / np.array(mu_current)
    return {
        "hp_cutoff_hz": float(hp_cutoff),
        "n_current_facetpy": n_current,
        "n_matlab_equivalent": n_matlab_equivalent,
        "n_legacy_upsampled_unit_reference": n_legacy_upsampled_unit,
        "mu_current_median": float(np.median(mu_current)),
        "mu_matlab_median": float(np.median(mu_matlab)),
        "mu_ratio_matlab_over_current_median": float(np.median(ratio)),
        "mu_current_min": float(np.min(mu_current)),
        "mu_current_max": float(np.max(mu_current)),
        "mu_matlab_min": float(np.min(mu_matlab)),
        "mu_matlab_max": float(np.max(mu_matlab)),
    }


def build_pipeline(probe: ParityProbe) -> Pipeline:
    """Build a CleanExJanik-style FACETpy pipeline with comparison checkpoints."""
    steps = [
        Loader(path=INPUT_FILE, preload=True),
        RawTransform("status_to_stim", _mark_status_as_stim),
        TriggerDetector(regex=TRIGGER_REGEX),
        CutAcquisitionWindow(),
        HighPassFilter(freq=1.0),
        UpSample(factor=UPSAMPLE),
        SliceAligner(ref_trigger_index=0),
        SubsampleAligner(ref_trigger_index=0),
        probe.checkpoint("alignment", to_native=True),
        AASCorrection(
            window_size=30,
            correlation_threshold=0.975,
            realign_after_averaging=True,
            apply_epoch_alpha_scaling=True,
        ),
        probe.checkpoint("aas", to_native=True),
        PCACorrection(
            n_components=0.95,
            hp_freq=1.0,
            exclude_channels=[30, 31],  # EMG/ECG in this dataset (0-based)
        ),
        probe.checkpoint("pca", to_native=True),
        DownSample(factor=UPSAMPLE),
        PasteAcquisitionWindow(),
        LowPassFilter(freq=70.0),
        probe.checkpoint("lowpass", to_native=False),
        ANCCorrection(use_c_extension=True),
        probe.checkpoint("anc", to_native=False),
        EDFExporter(path=OUTPUT_FILE, overwrite=True),
    ]
    return Pipeline(steps, name="CleanExJanik Equivalent (FACETpy)")


def render_markdown(
    stage_results: dict[str, StageComparison],
    anc_mu: dict[str, Any],
) -> str:
    """Create a markdown report with findings and improvement suggestions."""
    order = ["alignment", "aas", "pca", "lowpass", "anc"]

    lines = [
        "# CleanExJanik Parity Findings",
        "",
        "## Setup",
        "",
        f"- Input: `{INPUT_FILE}`",
        f"- Pipeline run mode: `channel_sequential=True`",
        "- Console mode: `classic`",
        f"- MATLAB references: `{', '.join(Path(REFERENCE_FILES[s]).name for s in order)}`",
        "",
        "## Step-wise Comparison",
        "",
        "| Stage | Channels | Samples | MAE | RMSE | Rel. RMSE (%) | Median Corr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for stage in order:
        row = stage_results[stage]
        lines.append(
            f"| `{stage}` | {row.channels_compared} | {row.samples_compared} | "
            f"{row.mae:.6e} | {row.rmse:.6e} | {row.rel_rmse_percent:.3f} | "
            f"{(f'{row.median_corr:.3f}' if row.median_corr is not None else 'n/a')} |"
        )

    lines += [
        "",
        "## ANC Parameter Focus (`mu`)",
        "",
        f"- ANC HP cutoff: `{anc_mu.get('hp_cutoff_hz', 'n/a')}` Hz",
        f"- FACETpy ANC filter order `N`: `{anc_mu.get('n_current_facetpy', 'n/a')}`",
        f"- MATLAB-equivalent ANC filter order `N` (native sample units): "
        f"`{anc_mu.get('n_matlab_equivalent', 'n/a')}`",
        f"- Legacy reference (if `artifact_length` were still in upsampled units): "
        f"`{anc_mu.get('n_legacy_upsampled_unit_reference', 'n/a')}`",
    ]

    if "mu_current_median" in anc_mu:
        lines += [
            f"- FACETpy median `mu`: `{anc_mu['mu_current_median']:.6e}`",
            f"- MATLAB-equivalent median `mu`: `{anc_mu['mu_matlab_median']:.6e}`",
            f"- Ratio (`MATLAB/FACETpy`): `{anc_mu['mu_ratio_matlab_over_current_median']:.3f}x`",
        ]
    else:
        lines.append(f"- ANC mu diagnostics error: `{anc_mu.get('error', 'unknown')}`")

    lines += [
        "",
        "## Likely Reasons for Differences",
        "",
        "- `NiazyFMRI.set` stores `Status` as EEG, so FACETpy needs a remap (`Status -> STIM`) before trigger detection.",
        "- `artifact_length` scaling on `DownSample` is now fixed in this branch, so ANC runs in native sample units.",
        "- MATLAB uses explicit `FindMissingTriggers(40,21)` logic; FACETpy trigger handling path is different here.",
        "- Pre-filtering details differ (`PreFilterGaussHPFrequency` in MATLAB vs `HighPassFilter(freq=1.0)` here).",
        "- PCA/OBS internals differ between implementations (`OBSNumPCs='auto'` behavior is not 1:1 with FACETpy PCA).",
        "",
        "## Suggestions",
        "",
        "1. Keep the new `DownSample` metadata scaling fix protected by regression tests (already added in this branch).",
        "2. Add an explicit FACETpy processor option mirroring `FindMissingTriggers(volumes, slices)` for deterministic trigger parity.",
        "3. Add an optional Gaussian HP pre-filter mode to match MATLAB `PreFilterGaussHPFrequency`.",
        "4. Keep storing per-stage parity metrics in CI for this Niazy dataset to catch regressions early.",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    probe = ParityProbe()
    pipeline = build_pipeline(probe)
    result = pipeline.run(channel_sequential=True)
    result.print_summary()

    anc_mu = compute_anc_mu_diagnostics(result.context)
    stage_payload = {k: asdict(v) for k, v in probe.stages.items()}

    payload = {
        "stage_comparison": stage_payload,
        "anc_mu_diagnostics": anc_mu,
        "output_file": OUTPUT_FILE,
        "success": bool(result.success),
        "execution_time_s": float(result.execution_time),
    }
    COMPARISON_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_text = render_markdown(probe.stages, anc_mu)
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print(f"Saved corrected output: {OUTPUT_FILE}")
    print(f"Saved comparison JSON: {COMPARISON_JSON}")
    print(f"Saved findings report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
