"""The canonical FACETpy correction chain, in one place.

Every pipeline tool in this directory built its own step list, and each one
silently used the ``Loader`` default ``artifact_to_trigger_offset = 0.0``. That
default is wrong for this recording: the gradient artifact starts *before* the
trigger, so with offset 0 a slice of it falls outside every epoch window. The
cost was not subtle — FARM's residual was **59.5 µV RMS instead of 20.1 µV**, a
factor of three, and it was invisible because nothing compared the runs against
``examples/``.

This module therefore holds the chain once, with the constants read from
``examples/complete_pipeline_example.py``, so a tool cannot quietly diverge from
the reference again.

Where PCA goes — corrected
--------------------------
An earlier version of this module claimed the cascade was trained on a
FARM-corrected but *not* PCA-corrected signal, and therefore ran ``PCACorrection``
after the learned stage in every arm. That claim was wrong, and it is worth
recording why, because the code read as if it had been checked.

The training bundle is produced by
``tools/dataset_building/extract_niazy_aas_pca4_artifact.py``, whose whole point
is the extra stage::

    ... | FARMCorrection(window_size=30, correlation_threshold=0.9)
        | PCACorrection(n_components=4, hp_freq=300.0)

and ``spatiotemporal_builder`` then sets ``artifact_template = artifact``
*before* the failure-mode enrichment. So the template the cascade is handed —
the thing subtracted to form its input ``noisy - template`` — is
**FARM + PCA/OBS(4, 300 Hz)**, not FARM alone. Running the reference PCA after
the cascade gave the model an input with a stage missing and then applied a
*different* PCA on top.

Hence :func:`cascade_template_stage`, which reproduces the bundle's primary
correction exactly. The **direct** models take the noisy signal itself, so no
template stage applies to them and the reference cleanup PCA is fair.

``include_pca=False`` exists for the ablation that shows what the stage is worth,
not as a convenience.
"""

from __future__ import annotations

from pathlib import Path

from facet.core import Pipeline
from facet.correction import FARMCorrection, PCACorrection
from facet.io import Loader
from facet.preprocessing import (
    Crop, DownSample, DropChannels, HighPassFilter, LowPassFilter,
    TriggerAligner, TriggerDetector, UpSample,
)
from facet.preprocessing.alignment import SubsampleAligner

#: Read from examples/complete_pipeline_example.py — do not edit that file.
ARTIFACT_TO_TRIGGER_OFFSET = -0.005      # s; the artifact starts before the trigger
NON_EEG_CHANNELS = ["EKG", "EMG", "EOG", "ECG"]
CROP = (0, 162)                          # s
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
        Loader(path=str(input_path), preload=True,
               artifact_to_trigger_offset=ARTIFACT_TO_TRIGGER_OFFSET),
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


def build(input_path: str | Path, *, correctors: list | None = None,
          include_pca: bool = True, name: str = "reference",
          trigger_regex: str = TRIGGER_REGEX, include_lowpass: bool = True) -> Pipeline:
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
    return [FARMCorrection(**TRAINING_FARM_KWARGS),
            PCACorrection(**TRAINING_PCA_KWARGS)]
