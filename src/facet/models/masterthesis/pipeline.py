"""Recorded correction pipeline used for the thesis comparisons.

The evaluation chain and the training-template chain have distinct FARM and PCA
settings. Keep them separate when comparing a direct model and a residual model.
All input paths are supplied by the caller.
"""

from __future__ import annotations

from pathlib import Path

from facet.core import Pipeline
from facet.correction import FARMCorrection, PCACorrection
from facet.io import Loader
from facet.preprocessing import (
    Crop,
    DownSample,
    DropChannels,
    HighPassFilter,
    LowPassFilter,
    TriggerAligner,
    TriggerDetector,
    UpSample,
)
from facet.preprocessing.alignment import SubsampleAligner

#: Read from examples/complete_pipeline_example.py — do not edit that file.
ARTIFACT_TO_TRIGGER_OFFSET = -0.005  # s; the artifact starts before the trigger
NON_EEG_CHANNELS = ["EKG", "EMG", "EOG", "ECG"]
CROP = (0, 162)  # s
UPSAMPLE = 10
FARM_KWARGS = dict(window_size=30, correlation_threshold=0.975, realign_after_averaging=True)
PCA_KWARGS = dict(n_components=0.95, hp_freq=70.0)

#: The primary correction of the *training bundle*, read from
#: tools/dataset_building/extract_niazy_aas_pca4_artifact.py. Both values differ
#: from the reference chain above (0.9 vs 0.975; PCA(4, 300) vs PCA(0.95, 70)),
#: so they are separate constants rather than shared ones.
TRAINING_FARM_KWARGS = dict(window_size=30, correlation_threshold=0.9)
TRAINING_PCA_KWARGS = dict(n_components=4, hp_freq=300.0)
LOWPASS_HZ = 70.0
HIGHPASS_HZ = 1.0
TRIGGER_REGEX = r"\b1\b"


def preprocessing(input_path: str | Path, *, trigger_regex: str = TRIGGER_REGEX) -> list:
    """Everything up to and including sub-sample trigger alignment."""
    return [
        Loader(path=str(input_path), preload=True, artifact_to_trigger_offset=ARTIFACT_TO_TRIGGER_OFFSET),
        DropChannels(channels=NON_EEG_CHANNELS, on_missing="ignore"),
        Crop(tmin=CROP[0], tmax=CROP[1]),
        HighPassFilter(freq=HIGHPASS_HZ),
        TriggerDetector(regex=trigger_regex),
        UpSample(factor=UPSAMPLE),
        TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False),
        SubsampleAligner(),
    ]


def postprocessing(*, include_pca: bool = True, include_lowpass: bool = True) -> list:
    """Residual cleanup and return to the native rate.

    ``include_lowpass=False`` exposes the exact pre-final-filter signal state
    for diagnostic plots; normal correction pipelines retain the 70-Hz filter.
    """
    steps: list = []
    if include_pca:
        steps.append(PCACorrection(**PCA_KWARGS))
    steps += [DownSample(factor=UPSAMPLE)]
    if include_lowpass:
        steps.append(LowPassFilter(freq=LOWPASS_HZ))
    return steps


def build(
    input_path: str | Path,
    *,
    correctors: list | None = None,
    include_pca: bool = True,
    name: str = "reference",
    trigger_regex: str = TRIGGER_REGEX,
    include_lowpass: bool = True,
) -> Pipeline:
    """Full chain with the given correction stages between pre- and postprocessing.

    ``correctors=[]`` gives the uncorrected reference through an otherwise
    identical chain, which is the only fair thing to measure a corrector against.
    """
    steps = preprocessing(input_path, trigger_regex=trigger_regex)
    steps += list(correctors or [])
    steps += postprocessing(include_pca=include_pca, include_lowpass=include_lowpass)
    return Pipeline(steps, name=name)


def farm() -> list:
    return [FARMCorrection(**FARM_KWARGS)]


def cascade_template_stage() -> list:
    """The primary correction the cascade's template was built from.

    Use this — not :func:`farm` — in front of a FARM-residual model, so the
    signal reaching the model is the one it was trained on. The two differ in
    both stages, not only in the added PCA.
    """
    return [FARMCorrection(**TRAINING_FARM_KWARGS), PCACorrection(**TRAINING_PCA_KWARGS)]
