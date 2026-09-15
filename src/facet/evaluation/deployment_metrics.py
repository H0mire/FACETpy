"""Metrics for judging a corrector on a real recording, without a clean reference.

Everything in :mod:`facet.evaluation.metrics` either needs a reference interval
outside the acquisition window or compares whole-signal statistics. Both are
useful, and both missed two failure modes that only appeared when fourteen
trained models were run end to end on an EDF:

**Residual gradient artifact.** Residual RMS answers "how much signal is left",
not "how much artifact is left" — and those come apart badly. A corrector that
removes the EEG along with the artifact scores *best* on residual RMS. The
gradient artifact is epoch-periodic, so the power sitting on the epoch-repetition
frequency and its harmonics is the part the corrector failed to remove, and it is
measurable without any clean signal.
:class:`GradientArtifactResidualCalculator` reports it in µV rather than as a
share of total power, because a share is misleading in exactly the case that
matters: an arm that deleted the EEG has almost no power left, so nearly all of
the little that remains is epoch-periodic and its share looks catastrophic while
its amount is small.

**Steps at the epoch seams.** Learned correctors work epoch by epoch and most of
them remove each segment's own mean. Reassembled into a continuous recording the
segments no longer share a baseline and the signal steps at every join. A
per-epoch evaluation on prepared tensors cannot see this at all — each epoch is
scored alone — so it does not show up until deployment.
:class:`EpochSeamStepCalculator` measures the jump across the epoch boundary
against the ordinary sample-to-sample jump, which makes it comparable between
recordings and sampling rates.

Both are descriptive. On a real recording there is no ground truth, so neither
says whether the *right* thing was removed; read them together with an accuracy
measure on data with a known clean signal.
"""

from __future__ import annotations

import mne
import numpy as np
from loguru import logger

from facet.core import ProcessingContext, Processor, ProcessorValidationError


def _epoch_rate_hz(triggers: np.ndarray, sfreq: float) -> float:
    """Repetition rate of the gradient artifact, from the trigger spacing.

    The median spacing, not the mean: a missing or doubled trigger shifts a mean
    enough to move the comb off the harmonics entirely, and then the metric
    silently reports background power instead of artifact.
    """
    if triggers.size < 2:
        raise ValueError("need at least two triggers to determine the epoch rate")
    return float(sfreq) / float(np.median(np.diff(np.sort(triggers))))


class GradientArtifactResidualCalculator(Processor):
    """Residual gradient artifact in µV, measured on the epoch harmonics.

    Needs no clean reference: the artifact repeats with the volume trigger, so
    its leftover power sits on that frequency and its multiples while brain
    activity does not concentrate there.

    Parameters
    ----------
    fmax : float, optional
        Highest harmonic to include, in Hz (default: 70.0, the band the
        reference chain keeps).
    tolerance_bins : float, optional
        Half-width of each harmonic in FFT bins (default: 1.5). The epoch period
        is not an integer number of samples, so the comb lines are not exactly
        on-grid.
    background_bins : float, optional
        Outer half-width of the guard region used to estimate the local
        background, in FFT bins (default: 12.0). Broadband activity also falls
        into the comb bins, and without subtracting it the metric carries a floor
        proportional to the broadband level — which would make a corrector that
        keeps the EEG look like it left more artifact than one that deleted it.
        Set to ``0`` to disable and measure raw comb power.
    tmin, tmax : float, optional
        Analysis window in seconds. Defaults to the whole recording; pass the
        steady-state part to keep the scan-onset transient out.

    Examples
    --------
    ::

        calc = GradientArtifactResidualCalculator(tmin=30.0)
        context = calc.execute(context)
        print(context.metadata.custom["gradient_artifact_residual"]["comb_rms_uv"])
    """

    name = "gradient_artifact_residual_calculator"
    description = "Residual gradient artifact on the epoch harmonics"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        fmax: float = 70.0,
        tolerance_bins: float = 1.5,
        background_bins: float = 12.0,
        tmin: float | None = None,
        tmax: float | None = None,
        verbose: bool = False,
    ) -> None:
        self.fmax = float(fmax)
        self.tolerance_bins = float(tolerance_bins)
        self.background_bins = float(background_bins)
        self.tmin = tmin
        self.tmax = tmax
        self.verbose = verbose
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        triggers = context.get_triggers()
        if triggers is None or len(triggers) < 2:
            raise ProcessorValidationError("Need at least two triggers to determine the epoch repetition rate.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw()
        sfreq = float(raw.info["sfreq"])
        triggers = np.asarray(context.get_triggers(), dtype=np.int64)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        if len(picks) == 0:
            logger.warning("No EEG channels found; skipping gradient-artifact residual")
            return context

        i0 = 0 if self.tmin is None else int(self.tmin * sfreq)
        i1 = raw.n_times if self.tmax is None else int(self.tmax * sfreq)
        data = raw.get_data(picks=picks)[:, i0:i1] * 1e6

        f_epoch = _epoch_rate_hz(triggers, sfreq)
        window = np.hanning(data.shape[1])
        psd = np.abs(np.fft.rfft(data * window, axis=-1)) ** 2
        freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)
        df = float(freqs[1] - freqs[0])

        mask = np.zeros(freqs.size, dtype=bool)
        excess = 0.0
        harmonic = 1
        while harmonic * f_epoch < self.fmax:
            offset = np.abs(freqs - harmonic * f_epoch)
            peak = offset <= self.tolerance_bins * df
            mask |= peak
            if not np.any(peak):
                harmonic += 1
                continue
            if self.background_bins > self.tolerance_bins:
                guard = (offset > self.tolerance_bins * df) & (offset <= self.background_bins * df)
                if np.any(guard):
                    # Median of the guard bins, divided by ln 2: periodogram bins
                    # of a noise process are exponentially distributed, whose
                    # median is ln(2) times its mean. Taking the median directly
                    # would under-estimate the background by 31 % and leave that
                    # much of the noise floor in the result; taking the mean would
                    # let a neighbouring line inflate it.
                    floor = np.median(psd[:, guard], axis=1, keepdims=True) / np.log(2.0)
                    # Clip per harmonic, not per bin: subtracting a background
                    # from single bins and clipping keeps every positive noise
                    # excursion, which is a floor of its own. Summing the bins
                    # first averages those excursions out.
                    band = psd[:, peak].sum(axis=1, keepdims=True)
                    excess += float(np.clip(band - int(peak.sum()) * floor, 0.0, None).sum())
                    harmonic += 1
                    continue
            excess += float(psd[:, peak].sum())
            harmonic += 1
        n_harmonics = harmonic - 1

        total = float(psd.sum())
        share = excess / max(total, 1e-30)
        rms = float(np.sqrt(np.mean(data**2)))
        result = {
            "comb_rms_uv": rms * float(np.sqrt(share)),
            "comb_share_pct": 100.0 * share,
            "comb_share_raw_pct": 100.0 * float(psd[:, mask].sum()) / max(total, 1e-30),
            "background_bins": self.background_bins,
            "rms_uv": rms,
            "epoch_rate_hz": f_epoch,
            "n_harmonics": n_harmonics,
            "fmax_hz": self.fmax,
            "window_s": [i0 / sfreq, i1 / sfreq],
            "n_channels": int(len(picks)),
        }
        logger.info(
            "Residual gradient artifact: {:.2f} µV ({:.2f} % of {:.2f} µV RMS), {} harmonics of {:.4f} Hz",
            result["comb_rms_uv"],
            result["comb_share_pct"],
            rms,
            n_harmonics,
            f_epoch,
        )
        if self.verbose:
            logger.info(
                "Gradient-artifact diagnostics: window {}, channels {}", result["window_s"], result["n_channels"]
            )

        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault("metrics", {})
        metrics["gradient_artifact_residual_uv"] = result["comb_rms_uv"]
        metrics["gradient_artifact_residual_share_pct"] = result["comb_share_pct"]
        new_metadata.custom["gradient_artifact_residual"] = result
        return context.with_metadata(new_metadata)


class EpochSeamStepCalculator(Processor):
    """Discontinuity at the epoch joins, relative to the ordinary sample step.

    A corrector that works per epoch and removes each segment's own mean leaves
    the segments without a common baseline; the reassembled signal then steps at
    every trigger. The ratio below is 1.0 for a corrector that leaves no seam and
    grows with the size of the step.

    The uncorrected signal also scores above 1.0 — the artifact itself jumps at
    the epoch boundary — so the reference point is a working corrector on the
    same recording, not the value 1.0 in the abstract.

    Parameters
    ----------
    tmin, tmax : float, optional
        Analysis window in seconds (default: the whole recording).

    Examples
    --------
    ::

        calc = EpochSeamStepCalculator(tmin=30.0)
        context = calc.execute(context)
        print(context.metadata.custom["epoch_seam_step"]["ratio"])
    """

    name = "epoch_seam_step_calculator"
    description = "Signal discontinuity at the epoch boundaries"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    #: Minimum number of usable seams. Below this the median is not meaningful
    #: and the metric reports NaN rather than a number nobody should trust.
    MIN_SEAMS = 8

    def __init__(self, tmin: float | None = None, tmax: float | None = None, verbose: bool = False) -> None:
        self.tmin = tmin
        self.tmax = tmax
        self.verbose = verbose
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw()
        sfreq = float(raw.info["sfreq"])
        triggers = np.asarray(context.get_triggers(), dtype=np.int64)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        if len(picks) == 0:
            logger.warning("No EEG channels found; skipping epoch-seam step")
            return context

        i0 = 0 if self.tmin is None else int(self.tmin * sfreq)
        i1 = raw.n_times if self.tmax is None else int(self.tmax * sfreq)
        data = raw.get_data(picks=picks)[:, i0:i1] * 1e6

        seams = triggers[(triggers > i0) & (triggers < i1 - 1)] - i0
        if seams.size < self.MIN_SEAMS:
            result = {
                "ratio": float("nan"),
                "seam_step_uv": float("nan"),
                "sample_step_uv": float("nan"),
                "n_seams": int(seams.size),
                "note": f"fewer than {self.MIN_SEAMS} seams in the window",
            }
        else:
            seam_step = float(np.median(np.abs(data[:, seams] - data[:, seams - 1])))
            sample_step = float(np.median(np.abs(np.diff(data, axis=1))))
            result = {
                "ratio": seam_step / max(sample_step, 1e-30),
                "seam_step_uv": seam_step,
                "sample_step_uv": sample_step,
                "n_seams": int(seams.size),
                "window_s": [i0 / sfreq, i1 / sfreq],
                "n_channels": int(len(picks)),
            }
            logger.info(
                "Epoch-seam step: {:.2f} µV against {:.2f} µV per sample -> ratio {:.2f} over {} seams",
                seam_step,
                sample_step,
                result["ratio"],
                seams.size,
            )

        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault("metrics", {})
        metrics["epoch_seam_step_ratio"] = result["ratio"]
        new_metadata.custom["epoch_seam_step"] = result
        return context.with_metadata(new_metadata)
