"""The pre-registered criterion, checked against the run it was written from.

The point of fixing thresholds before the deployment editions are measured is
that they cannot then be tuned to the answer. The way to keep that honest is to
pin what the criterion says about the *old* results, so a later edit that quietly
moves a gate shows up as a failing test rather than as a better-looking table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from facet.evaluation.correction_verdict import (
    FARM_UV,
    UNCORRECTED_UV,
    Verdict,
    verdict,
)

RUN6 = Path("output/pipeline_demo/family_stack/arm_diagnosis.json")


@pytest.mark.unit
def test_no_learned_family_was_farm_comparable_in_run6():
    """The claim the images support: three reduced the artifact, none corrected."""
    for residual, band, seam in [(11.88, 0.87, 2.22), (12.69, 1.61, 1.28),
                                 (15.87, 1.78, 1.97)]:
        assert not verdict(residual, band, seam).farm_comparable


@pytest.mark.unit
def test_the_reference_arms_are_farm_comparable():
    assert verdict(4.62, 1.00, 1.38).farm_comparable      # FARM itself
    assert verdict(5.05, 1.20, 0.96).farm_comparable      # the Weg-A cascade


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, residual, band, seam",
    [
        ("vit_spectrogram deleted the signal", 0.38, 0.26, 5.72),
        ("denoise_mamba deleted the signal", 2.55, 0.28, 2.18),
        ("st_gnn stepped at every seam", 8.63, 0.56, 6.20),
        ("cascaded_dae added energy", 42.92, 7.77, 3.17),
        ("dhct_gan diverged", 1326.72, 4980.95, 17.16),
        ("the uncorrected arm", UNCORRECTED_UV, 147.46, 4.97),
    ],
)
def test_the_run6_failures_fail(name, residual, band, seam):
    assert not verdict(residual, band, seam).artifact_reduced, name


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, residual, band, seam",
    [
        ("ic_unet", 11.88, 0.87, 2.22),
        ("demucs", 12.69, 1.61, 1.28),
        ("conv_tasnet", 15.87, 1.78, 1.97),
    ],
)
def test_three_of_the_fourteen_did_correct(name, residual, band, seam):
    """Worth pinning because it corrects the project's own shorthand.

    "None of the fourteen corrected" was the working summary and it is too
    strong. These three removed 94-95 % of what the uncorrected arm left and kept
    the EEG band; they are simply worse than FARM. The criterion separates the
    two claims, which is what it is for.
    """
    v = verdict(residual, band, seam)
    assert v.artifact_reduced, name
    assert not v.farm_comparable, f"{name} is not within 2x of FARM"


@pytest.mark.unit
def test_a_deleted_signal_cannot_pass_on_a_small_residual_alone():
    """The trap the whole criterion exists to close: removing the EEG scores a
    tiny residual, which on a residual-only test looks like the best result."""
    deleted = verdict(0.10, 0.05, 1.0)
    assert not deleted.artifact_reduced
    assert any("gelöscht" in r for r in deleted.reasons)


@pytest.mark.unit
def test_a_missing_seam_measurement_is_not_a_pass():
    assert not verdict(5.0, 1.0, float("nan")).artifact_reduced


@pytest.mark.unit
def test_references_can_be_overridden_for_another_recording():
    """Thresholds are relative to *this* run's own uncorrected and FARM arms."""
    assert verdict(20.0, 1.0, 1.0, uncorrected_uv=1000.0).artifact_reduced
    assert not verdict(20.0, 1.0, 1.0, uncorrected_uv=40.0).artifact_reduced


@pytest.mark.unit
def test_verdict_reads_as_a_sentence():
    v = verdict(4.62, 1.00, 1.38)
    assert isinstance(v, Verdict)
    assert str(v).startswith("FARM-vergleichbar")
    assert len(v.reasons) == 3


@pytest.mark.unit
@pytest.mark.skipif(not RUN6.exists(), reason="run-6 diagnosis not in this checkout")
def test_against_the_stored_run6_diagnosis():
    """Same verdicts, computed from the file rather than from numbers typed here."""
    rows = json.loads(RUN6.read_text())["rows"]
    by_arm = {r["arm"]: r for r in rows}
    corrects = {
        arm for arm, r in by_arm.items()
        if verdict(r["comb_rms_uv"], r["eeg_band_rel_farm"],
                   r["epoch_boundary_step_ratio"]).artifact_reduced
    }
    learned = corrects - {"farm", "farm_pca4", "wega_cascade", "cascade_spk0",
                          "cascade_spk1", "wega_direct", "uncorrected", "legacy_dl"}
    assert learned == {"ic_unet", "demucs", "conv_tasnet"}
    assert verdict(*[by_arm["farm"][k] for k in
                     ("comb_rms_uv", "eeg_band_rel_farm",
                      "epoch_boundary_step_ratio")]).farm_comparable


@pytest.mark.unit
def test_the_gates_survive_a_change_of_analysis_window():
    """The same arm, measured over 5.5 s and over 130 s, must get the same verdict.

    The residual in µV is a property of the estimator's window, not of the signal:
    FARM measures 4.62 µV over 5.5 s and 0.60 µV over 130 s. Passing the run's own
    reference arms is what makes the criterion invariant to that — and forgetting
    to would move every threshold by a factor of seven.
    """
    short = verdict(4.62, 1.00, 1.38, uncorrected_uv=259.24, farm_uv=4.62)
    long = verdict(0.60, 1.00, 1.38, uncorrected_uv=36.43, farm_uv=0.60)
    assert short.farm_comparable and long.farm_comparable

    # conv_tasnet: the same 3.4x-worse-than-FARM arm in both windows.
    short_ct = verdict(15.87, 1.78, 1.97, uncorrected_uv=259.24, farm_uv=4.62)
    long_ct = verdict(15.87 * 0.14, 1.78, 1.97, uncorrected_uv=36.43, farm_uv=0.60)
    assert short_ct.artifact_reduced and long_ct.artifact_reduced
    assert not short_ct.farm_comparable and not long_ct.farm_comparable
