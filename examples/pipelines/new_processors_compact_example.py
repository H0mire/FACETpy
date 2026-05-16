"""
Compact demo of all newly added MATLAB-parity processors.

Audience: researchers who want a practical starting point.

What this file demonstrates
---------------------------
Preprocessing/reporting processors:
- AnalyzeDataReport
- CheckDataReport
- MATLABPreFilter
- MissingTriggerCompleter
- SliceTriggerGenerator

Correction processors:
- VolumeArtifactCorrection
- FARMCorrection
- CorrespondingSliceCorrection
- VolumeTriggerCorrection
- MoosmannCorrection

Usage
-----
1) Run one or more demos by setting DEMO_CHOICES below (by default, all demos run).
2) Start with "farm_volume" first.
3) Outputs are written to ./output/new_processors_compact/
"""

from __future__ import annotations

from pathlib import Path

from facet import (
    AnalyzeDataReport,
    CheckDataReport,
    CorrespondingSliceCorrection,
    CutAcquisitionWindow,
    DownSample,
    EDFExporter,
    FARMCorrection,
    Loader,
    LowPassFilter,
    MATLABPreFilter,
    MissingTriggerCompleter,
    MoosmannCorrection,
    PasteAcquisitionWindow,
    Pipeline,
    RawTransform,
    SliceAligner,
    SliceTriggerGenerator,
    SubsampleAligner,
    TriggerDetector,
    UpSample,
    VolumeArtifactCorrection,
    VolumeTriggerCorrection,
)

# ---------------------------------------------------------------------------
# Simple configuration (safe defaults for examples/datasets/NiazyFMRI.set)
# ---------------------------------------------------------------------------
INPUT_FILE = "./examples/datasets/NiazyFMRI.set"
MOTION_FILE = "./examples/datasets/headmotiondata_829s.tsv"
OUTPUT_DIR = Path("./output/new_processors_compact")

TRIGGER_REGEX = r"\b1\b"
VOLUMES = 40
SLICES_PER_VOLUME = 21
UPSAMPLE = 10

# Choose what to run (you can run one or many).
DEMO_CHOICES = {
    "trigger_utilities",   # MissingTriggerCompleter + SliceTriggerGenerator
    "farm_volume",         # VolumeArtifactCorrection + FARMCorrection
    "corresponding_slice", # CorrespondingSliceCorrection
    "volume_trigger",      # VolumeTriggerCorrection
    "moosmann",            # MoosmannCorrection
}


def _status_to_stim(raw):
    """Map EEGLAB 'Status' channel to STIM for trigger detection."""
    out = raw.copy()
    if "Status" in out.ch_names:
        out.set_channel_types({"Status": "stim"})
    return out


def _base_steps():
    """Shared preprocessing steps used by all correction demos."""
    return [
        Loader(path=INPUT_FILE, preload=True),
        RawTransform("status_to_stim", _status_to_stim),
        TriggerDetector(regex=TRIGGER_REGEX),
        AnalyzeDataReport(),
        CheckDataReport(require_triggers=True, strict=False),
        MissingTriggerCompleter(
            volumes=VOLUMES,
            slices=SLICES_PER_VOLUME,
            strict=False,
            add_annotations=False,
        ),
        CutAcquisitionWindow(),
        MATLABPreFilter(lp_frequency=70.0, hp_frequency=1.0, gauss_hp_frequency=1.0),
        UpSample(factor=UPSAMPLE),
        SliceAligner(ref_trigger_index=0),
        SubsampleAligner(ref_trigger_index=0),
    ]


def run_trigger_utilities_demo() -> None:
    """Show MissingTriggerCompleter and SliceTriggerGenerator in isolation."""
    print("\n[trigger_utilities] running...")

    # Run only through trigger completion first.
    pipeline = Pipeline(
        [
            Loader(path=INPUT_FILE, preload=True),
            RawTransform("status_to_stim", _status_to_stim),
            TriggerDetector(regex=TRIGGER_REGEX),
            MissingTriggerCompleter(
                volumes=VOLUMES,
                slices=SLICES_PER_VOLUME,
                strict=False,
                add_annotations=False,
            ),
        ],
        name="Trigger utilities",
    )
    result = pipeline.run(channel_sequential=True)
    ctx = result.context
    if ctx is None or not ctx.has_triggers():
        print("  no triggers available, skipping SliceTriggerGenerator demo")
        return

    # Emulate a volume-trigger recording by keeping every N-th trigger,
    # then regenerate slice triggers with SliceTriggerGenerator.
    volume_triggers = ctx.get_triggers()[::SLICES_PER_VOLUME]
    vol_ctx = ctx.with_triggers(volume_triggers)
    vol_ctx = SliceTriggerGenerator(
        slices=SLICES_PER_VOLUME,
        duration_samples=ctx.get_artifact_length(),
        relative_position=0.0,
        add_annotations=False,
    ).execute(vol_ctx)

    print(f"  original slice triggers: {len(ctx.get_triggers())}")
    print(f"  emulated volume triggers: {len(volume_triggers)}")
    print(f"  regenerated slice triggers: {len(vol_ctx.get_triggers())}")


def run_correction_demo(name: str, correction_steps: list, output_name: str) -> None:
    """Run one correction demo pipeline and export an EDF result."""
    print(f"\n[{name}] running...")
    out_file = str(OUTPUT_DIR / output_name)

    pipeline = Pipeline(
        _base_steps()
        + correction_steps
        + [
            DownSample(factor=UPSAMPLE),
            PasteAcquisitionWindow(),
            LowPassFilter(freq=70.0),
            EDFExporter(path=out_file, overwrite=True),
        ],
        name=name,
    )

    result = pipeline.run(channel_sequential=True)
    result.print_summary()

    ctx = result.context
    if ctx is not None:
        print(f"  output: {out_file}")
        print(f"  metadata.custom keys: {sorted(ctx.metadata.custom.keys())}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if "trigger_utilities" in DEMO_CHOICES:
        run_trigger_utilities_demo()

    if "farm_volume" in DEMO_CHOICES:
        run_correction_demo(
            "FARM + Volume Artifact",
            [
                VolumeArtifactCorrection(template_count=5, weighting_position=0.8, weighting_slope=20.0),
                FARMCorrection(window_size=30, correlation_threshold=0.9),
            ],
            "corrected_farm_volume.edf",
        )

    if "corresponding_slice" in DEMO_CHOICES:
        run_correction_demo(
            "Corresponding Slice",
            [
                CorrespondingSliceCorrection(
                    slices_per_volume=SLICES_PER_VOLUME,
                    window_size=30,
                )
            ],
            "corrected_corresponding_slice.edf",
        )

    if "volume_trigger" in DEMO_CHOICES:
        run_correction_demo(
            "Volume Trigger",
            [VolumeTriggerCorrection(window_size=30)],
            "corrected_volume_trigger.edf",
        )

    if "moosmann" in DEMO_CHOICES:
        run_correction_demo(
            "Moosmann",
            [
                MoosmannCorrection(
                    rp_file=MOTION_FILE,
                    window_size=30,
                    motion_threshold=5.0,
                )
            ],
            "corrected_moosmann.edf",
        )


if __name__ == "__main__":
    main()
