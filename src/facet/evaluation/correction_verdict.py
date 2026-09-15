"""Does an arm actually correct? A criterion fixed before the numbers arrive.

Run 6 ran fourteen trained model families end to end on a real recording and not
one of them corrected. The retrained deployment editions exist to answer whether
that was the architectures or the objective — and that question is only
answerable if "corrected" means something decided in advance. Choosing a
threshold after seeing the results is choosing which models pass.

So the thresholds below are written down **before** the deployment editions were
measured through a pipeline, and every one of them is derived from a number
already in the run-6 record rather than invented here:

======================  ===========  =====================================================
gate                    threshold    where the number comes from
======================  ===========  =====================================================
artifact removed        <= 64.8 µV   a quarter of the uncorrected arm's 259.24 µV
signal kept             0.5 - 2.0x   the band FARM's own arms sat in; below 0.5 the run-6
                                     diagnosis called it "EEG-Band gelöscht", above 2.0
                                     "Energie hinzugefügt statt entfernt"
continuous              <= 2.5       the seam-step ratio the run-6 diagnosis already used
                                     to flag "Stufen an den Epochennähten"
======================  ===========  =====================================================

All three must hold. What they establish is that an arm **reduced the artifact
without destroying the signal** — and that is all they establish. The label for
passing them is ``artifact_reduced``, not "corrected", and the rename is the
result of looking at the traces rather than at the numbers.

**Why the rename.** The gates were first written with ``corrects`` as the label,
and on the run-6 data ``ic_unet`` (11.88 µV), ``demucs`` (12.69 µV) and
``conv_tasnet`` (15.87 µV) passed. Plotting five seconds of the scan for each arm
showed what those numbers were hiding: all three leave a **regular spike train at
the epoch rate** — a residual gradient artifact that is the dominant feature of
the trace, reaching ±60 µV in ``demucs`` against FARM's ±30 µV of EEG. Nobody
looking at those traces would call them corrected.

The metric was right and the threshold was wrong. 25 % of the uncorrected arm is
64.8 µV, which no useful correction is anywhere near. The arms that *look* like
FARM are exactly those within about a factor of two of its residual — FARM 4.62,
FARM+PCA 4.61, the Weg-A cascade 5.05, the spike-aware cascades 3.58 and 5.47 —
and the first arm that looks broken is ``st_gnn`` at 8.63. :data:`FARM_COMPARABLE_UV`
was already set at 2x FARM before that check, and the images put a line under it.

The looser gate is kept because it still separates "reduced the artifact" from
"made things worse", which is a real distinction across the fourteen families.
It is simply not the distinction its old name claimed.

**Always pass this run's own reference arms.** The gates are ratios for a reason:
the residual figure in µV is not a property of the signal alone, it is a property
of the estimator's window. Measured on the same two arms over two windows of the
same recording:

=============  ==========  ==========  =========
arm            5.5 s       130 s       factor
=============  ==========  ==========  =========
uncorrected    259.24 µV   36.43 µV    0.14
FARM             4.62 µV    0.60 µV    0.13
=============  ==========  ==========  =========

Both fall by the same factor, so the *ratio* between them is stable (56.1 against
60.7) while the absolute numbers are not — the comb's tolerance is measured in
FFT bins, and a 24x longer window makes each bin 24x narrower in Hz, so it
catches proportionally less broadband background. Every gate here is therefore
expressed against ``uncorrected_uv`` and ``farm_uv``, and the module-level
defaults below belong to **one specific measurement**: the run-6 pipeline on
``NiazyFMRI.edf`` with a 5.5 s analysis window. Using them on a run analysed over
the full acquisition would be wrong by a factor of seven.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Residual gradient artifact of the uncorrected arm in run 6, in µV, **over a
#: 5.5 s analysis window**. The arm carried no corrector at all — not even the
#: cleanup PCA, which removes 16.7 % of the raw power and made every earlier
#: removal figure look smaller than it was. Over the full 130 s the same arm
#: measures 36.43 µV; see the module docstring before reusing this default.
UNCORRECTED_UV = 259.24

#: FARM's residual in the same run and the same window, the reference a corrector
#: is measured against. 0.60 µV over the full acquisition.
FARM_UV = 4.62

#: An arm has to remove at least three quarters of what the uncorrected arm left.
REMOVED_FRACTION = 0.25

#: Band the EEG-band power must stay in, relative to FARM.
BAND_MIN, BAND_MAX = 0.5, 2.0

#: Above this the reassembled signal steps at the epoch joins.
SEAM_MAX = 2.5

#: Separate, harder label: within a factor of two of FARM.
FARM_COMPARABLE_UV = 2.0 * FARM_UV


@dataclass(frozen=True)
class Verdict:
    """Outcome for one arm, with the reason spelled out rather than a bare bool."""

    #: Artifact reduced and signal kept. **Not** the same as "usable": see the
    #: module docstring for the traces that forced this field to be renamed.
    artifact_reduced: bool
    #: Within a factor of two of FARM's residual, which is the point at which the
    #: trace stops showing a periodic spike train.
    farm_comparable: bool
    reasons: tuple[str, ...]

    @property
    def corrects(self) -> bool:
        """Deprecated alias for :attr:`farm_comparable`.

        The old ``corrects`` meant ``artifact_reduced``, and that reading passed
        three arms whose traces are dominated by residual artifact. Anything that
        asks "did this correct?" should get the strict answer.
        """
        return self.farm_comparable

    def __str__(self) -> str:
        label = (
            "FARM-vergleichbar"
            if self.farm_comparable
            else "Artefakt reduziert, aber sichtbare Reste"
            if self.artifact_reduced
            else "korrigiert nicht"
        )
        return f"{label}: {'; '.join(self.reasons)}"


def verdict(
    residual_uv: float,
    eeg_band_rel_farm: float,
    seam_step_ratio: float,
    *,
    uncorrected_uv: float = UNCORRECTED_UV,
    farm_uv: float = FARM_UV,
) -> Verdict:
    """Judge one arm against the pre-registered gates.

    Parameters
    ----------
    residual_uv : float
        Residual gradient artifact on the epoch harmonics, from
        :class:`facet.evaluation.deployment_metrics.GradientArtifactResidualCalculator`.
    eeg_band_rel_farm : float
        Power in the EEG band divided by FARM's, on the same recording.
    seam_step_ratio : float
        From :class:`facet.evaluation.deployment_metrics.EpochSeamStepCalculator`.
    uncorrected_uv, farm_uv : float
        The same two reference arms measured in *this* run. Defaults are run 6's.

    Returns
    -------
    Verdict

    Examples
    --------
    ::

        >>> verdict(4.61, 1.00, 1.0).farm_comparable      # FARM + cleanup PCA
        True
        >>> verdict(15.87, 1.78, 1.97).artifact_reduced   # Conv-TasNet: yes ...
        True
        >>> verdict(15.87, 1.78, 1.97).farm_comparable    # ... but the trace is
        False                                             # a spike train
        >>> verdict(0.38, 0.26, 6.2).artifact_reduced     # the deleted-signal arm
        False
    """
    limit = REMOVED_FRACTION * uncorrected_uv
    reasons: list[str] = []

    removed = residual_uv <= limit
    reasons.append(
        f"Artefaktrest {residual_uv:.2f} µV "
        f"{'≤' if removed else '>'} {limit:.2f} µV ({REMOVED_FRACTION:.0%} von unkorrigiert)"
    )

    kept = BAND_MIN <= eeg_band_rel_farm <= BAND_MAX
    if eeg_band_rel_farm < BAND_MIN:
        reasons.append(f"EEG-Band {eeg_band_rel_farm:.2f}x FARM — Signal gelöscht")
    elif eeg_band_rel_farm > BAND_MAX:
        reasons.append(f"EEG-Band {eeg_band_rel_farm:.2f}x FARM — Energie hinzugefügt")
    else:
        reasons.append(f"EEG-Band {eeg_band_rel_farm:.2f}x FARM — erhalten")

    # A NaN seam ratio means too few seams to form a median, which is a missing
    # measurement, not a passing one.
    continuous = seam_step_ratio == seam_step_ratio and seam_step_ratio <= SEAM_MAX
    reasons.append(f"Nahtverhältnis {seam_step_ratio:.2f} {'≤' if continuous else '>'} {SEAM_MAX}")

    reduced = bool(removed and kept and continuous)
    return Verdict(
        artifact_reduced=reduced,
        farm_comparable=bool(reduced and residual_uv <= 2.0 * farm_uv),
        reasons=tuple(reasons),
    )
