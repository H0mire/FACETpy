"""Tests for the paired spike comparison and per-example metrics.

A superiority claim rests on these numbers, so the statistics get the same
treatment as the model code: known inputs with known answers.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from facet.training.spike_metrics import compute_spike_metrics, compute_spike_metrics_per_example

_SPEC = importlib.util.spec_from_file_location(
    "paired_spike_comparison", Path("tools/evaluation/paired_spike_comparison.py")
)
psc = importlib.util.module_from_spec(_SPEC)
sys.modules["paired_spike_comparison"] = psc
_SPEC.loader.exec_module(psc)


def _case(n=8, t=400, spike_at=200, amp=300e-6, seed=0):
    rng = np.random.default_rng(seed)
    clean = (rng.standard_normal((n, t)) * 5e-6).astype(np.float64)
    labels = np.zeros((n, t), dtype=bool)
    for i in range(n):
        clean[i, spike_at - 3 : spike_at + 4] += amp
        labels[i, spike_at - 3 : spike_at + 4] = True
    return clean, labels


# ---------------------------------------------------------------------------
# Per-example metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_per_example_averages_match_the_aggregate():
    """The per-example arrays must be the same numbers the averages come from."""
    clean, labels = _case()
    rng = np.random.default_rng(1)
    pred = clean + rng.standard_normal(clean.shape) * 20e-6

    agg = compute_spike_metrics(pred, clean, labels, neighborhood_samples=50)
    per = compute_spike_metrics_per_example(pred, clean, labels, neighborhood_samples=50)

    assert per["example_index"].size == int(agg["n_spike_examples"])
    assert np.nanmean(per["amplitude_ratio"]) == pytest.approx(agg["spike_amplitude_ratio"], rel=1e-9)
    assert np.nanmean(per["morphology_corr"]) == pytest.approx(agg["spike_morphology_corr"], rel=1e-9)
    assert np.nanmean(per["contrast_db"]) == pytest.approx(agg["spike_contrast_db"], rel=1e-9)


@pytest.mark.unit
def test_per_example_indices_identify_the_rows():
    clean, labels = _case(n=6)
    labels[2] = False          # example 2 carries no spike
    labels[4] = False
    per = compute_spike_metrics_per_example(clean, clean, labels, neighborhood_samples=50)
    assert per["example_index"].tolist() == [0, 1, 3, 5]


@pytest.mark.unit
def test_perfect_reconstruction_is_ideal_per_example():
    clean, labels = _case()
    per = compute_spike_metrics_per_example(clean.copy(), clean, labels, neighborhood_samples=50)
    assert np.allclose(per["amplitude_ratio"], 1.0)
    assert np.allclose(per["morphology_corr"], 1.0)
    assert np.allclose(per["latency_drift_samples"], 0.0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_wilcoxon_detects_a_consistent_shift_and_ignores_noise():
    consistent = np.full(20, 0.5)
    _, p_shift = psc.wilcoxon_signed_rank(consistent)
    rng = np.random.default_rng(0)
    _, p_noise = psc.wilcoxon_signed_rank(rng.standard_normal(20))
    assert p_shift < 0.001
    assert p_noise > 0.1


@pytest.mark.unit
def test_wilcoxon_returns_nan_below_six_pairs():
    """Too few pairs must not produce a confident-looking p-value."""
    _, p = psc.wilcoxon_signed_rank(np.array([1.0, 2.0, 3.0]))
    assert np.isnan(p)


@pytest.mark.unit
def test_hodges_lehmann_recovers_a_known_shift():
    assert psc.hodges_lehmann(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(1.0)
    assert psc.hodges_lehmann(np.array([-2.0, 0.0, 2.0])) == pytest.approx(0.0)


@pytest.mark.unit
def test_cliffs_delta_bounds():
    a, b = np.arange(10.0), np.arange(10.0) + 100
    assert psc.cliffs_delta(a, b) == pytest.approx(-1.0)
    assert psc.cliffs_delta(b, a) == pytest.approx(1.0)
    assert psc.cliffs_delta(a, a.copy()) == pytest.approx(0.0)


@pytest.mark.unit
def test_holm_is_monotone_and_never_below_raw():
    corrected = psc.holm({"a": 0.001, "b": 0.02, "c": 0.4}, alpha=0.05)
    assert corrected["a"]["p_holm"] >= corrected["a"]["p_raw"]
    assert corrected["a"]["p_holm"] <= corrected["b"]["p_holm"] <= corrected["c"]["p_holm"]


@pytest.mark.unit
def test_bootstrap_ci_brackets_the_effect():
    """On symmetric differences the HL estimate and the mean coincide."""
    rng = np.random.default_rng(2)
    diff = rng.standard_normal(200) + 3.0
    lo, hi = psc.bootstrap_ci(diff, n=300, seed=0, alpha=0.05)
    assert lo < psc.hodges_lehmann(diff) < hi
    assert lo < diff.mean() < hi


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_paired_run_reports_shared_examples_and_exclusions(tmp_path):
    """Arms must be matched by example index, with the mismatch reported."""
    path = tmp_path / "per_example.csv"
    keys = ["amplitude_ratio", "morphology_corr", "contrast_db",
            "latency_drift_samples", "neighborhood_snr_db", "peak_over_residual", "rmse_uv"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "example_index", *keys])
        for i in range(12):
            w.writerow(["model", i, 1.1, 0.4, 9.0, 0.2, 8.0, 3.0, 10.0])
        for i in range(2, 12):                      # arm b lacks examples 0 and 1
            w.writerow(["aas_ideal", i, 1.4, 0.2, 5.0, 0.9, 2.0, 5.0, 40.0])

    data, clusters = psc.load(path)
    # No spike_event_id column: every row becomes its own cluster, which keeps
    # the old behaviour available for files written before clustering existed.
    assert all(c.startswith("row") for c in clusters.values())
    shared = sorted(set(data["model"]) & set(data["aas_ideal"]))
    assert shared == list(range(2, 12))
    assert sorted(set(data["model"]) - set(data["aas_ideal"])) == [0, 1]
    # derived comparables exist and point the right way
    assert data["model"][5]["amplitude_ratio_abs_error"] == pytest.approx(0.1)
    assert data["aas_ideal"][5]["amplitude_ratio_abs_error"] == pytest.approx(0.4)


def _write_clustered(path, n_events, per_event, effect=2.0):
    """One CSV with ``per_event`` electrode replicates of each spike event.

    The replicates carry the same underlying difference plus a small per-replicate
    wobble, which is exactly the structure that makes a naive per-row test
    anti-conservative: the information content is ``n_events``, not
    ``n_events * per_event``.
    """
    keys = ["amplitude_ratio", "morphology_corr", "contrast_db",
            "latency_drift_samples", "neighborhood_snr_db", "peak_over_residual", "rmse_uv"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "example_index", "spike_event_id", "target_channel", *keys])
        idx = 0
        for event in range(n_events):
            for ch in range(per_event):
                wobble = 0.01 * ch
                w.writerow(["model", idx, f"e{event}", ch,
                            1.1, 0.4 + wobble, 9.0, 0.2, 8.0 + effect + wobble, 3.0, 10.0])
                w.writerow(["aas_ideal", idx, f"e{event}", ch,
                            1.4, 0.2 + wobble, 5.0, 0.9, 8.0 + wobble, 5.0, 40.0])
                idx += 1
    return path


@pytest.mark.unit
def test_load_reads_the_event_id_and_drops_the_bookkeeping_columns(tmp_path):
    """The cluster id must reach the caller and must not become a metric."""
    path = _write_clustered(tmp_path / "pe.csv", n_events=3, per_event=4)
    data, clusters = psc.load(path)
    assert set(clusters.values()) == {"e0", "e1", "e2"}
    assert len(clusters) == 12
    # spike_event_id / target_channel are structure, not measurements.
    assert "spike_event_id" not in data["model"][0]
    assert "target_channel" not in data["model"][0]


@pytest.mark.unit
def test_event_level_collapses_replicates(tmp_path, capsys):
    """n reported is the event count, not the row count."""
    path = _write_clustered(tmp_path / "pe.csv", n_events=8, per_event=5)
    out = tmp_path / "out"
    sys.argv = ["paired", "--per-example", str(path), "--arm-a", "model",
                "--arm-b", "aas_ideal", "--out", str(out), "--bootstrap", "200"]
    psc.main()
    rows = list(csv.DictReader((out / "paired_model_vs_aas_ideal.csv").open()))
    by_metric = {r["metric"]: r for r in rows}
    assert int(by_metric["neighborhood_snr_db"]["n_events"]) == 8
    assert int(by_metric["neighborhood_snr_db"]["n_paired_windows"]) == 40
    meta = json.loads((out / "paired_model_vs_aas_ideal.json").read_text())
    assert meta["n_independent_events"] == 8
    assert meta["windows_per_event"] == {f"e{i}": 5 for i in range(8)}


@pytest.mark.unit
def test_too_few_events_is_reported_as_untestable(tmp_path):
    """Two events must not borrow significance from their electrode replicates."""
    path = _write_clustered(tmp_path / "pe.csv", n_events=2, per_event=19)
    out = tmp_path / "out"
    sys.argv = ["paired", "--per-example", str(path), "--arm-a", "model",
                "--arm-b", "aas_ideal", "--out", str(out), "--bootstrap", "200"]
    psc.main()
    rows = list(csv.DictReader((out / "paired_model_vs_aas_ideal.csv").open()))
    row = next(r for r in rows if r["metric"] == "neighborhood_snr_db")
    assert int(row["n_events"]) == 2
    assert int(row["n_paired_windows"]) == 38
    assert row["event_testable"] == "False"
    assert row["significant"] == "False"
    # The window level still carries a p-value, and it is small — which is
    # precisely why it must not be the one that decides significance.
    assert float(row["window_p_raw"]) < 0.01


@pytest.mark.unit
def test_cluster_column_is_selectable(tmp_path):
    """A bulk table clusters by epoch_id and writes to its own filename."""
    path = tmp_path / "bulk.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "example_index", "epoch_id", "target_channel", "rmse_uv", "clean_snr_db", "has_spike"])
        idx = 0
        for epoch in range(9):
            for ch in range(3):
                w.writerow(["model", idx, f"ep{epoch}", ch, 10.0 + 0.1 * ch, 5.0 + 0.1 * ch, 0])
                w.writerow(["null_output", idx, f"ep{epoch}", ch, 30.0 + 0.1 * ch, 0.0, 0])
                idx += 1
    out = tmp_path / "out"
    sys.argv = ["paired", "--per-example", str(path), "--cluster-column", "epoch_id",
                "--arm-a", "model", "--arm-b", "null_output", "--out", str(out), "--bootstrap", "200"]
    psc.main()
    written = out / "paired_model_vs_null_output_epoch_id.csv"
    assert written.exists(), "the cluster column must not overwrite the spike-level output"
    row = next(r for r in csv.DictReader(written.open()) if r["metric"] == "rmse_uv")
    assert int(row["n_events"]) == 9
    assert float(row["event_hodges_lehmann_difference"]) == pytest.approx(-20.0, abs=1e-6)


@pytest.mark.unit
def test_bootstrap_interval_contains_its_own_estimator(tmp_path):
    """The interval must bracket the Hodges-Lehmann effect it is paired with.

    Bootstrapping the mean while reporting the HL median put the estimate on the
    interval boundary for skewed differences; the regression guards against that
    combination returning.
    """
    rng = np.random.default_rng(0)
    skewed = np.concatenate([rng.normal(1.0, 0.2, 40), rng.normal(9.0, 0.4, 6)])
    lo, hi = psc.bootstrap_ci(skewed, 500, seed=0, alpha=0.05)
    hl = psc.hodges_lehmann(skewed)
    assert lo <= hl <= hi


@pytest.mark.unit
def test_hl_bootstrap_is_refused_on_very_large_samples(tmp_path):
    """Cost guard: the O(n^2)-per-resample estimator must not be attempted at scale."""
    big = np.arange(psc.HL_BOOTSTRAP_MAX_N + 1, dtype=float)
    lo, hi = psc.bootstrap_ci(big, 10, seed=0, alpha=0.05)
    assert np.isnan(lo) and np.isnan(hi)
    # The mean-based interval stays available for the descriptive window level.
    lo_mean, hi_mean = psc.bootstrap_ci(big, 50, seed=0, alpha=0.05, statistic="mean")
    assert np.isfinite(lo_mean) and np.isfinite(hi_mean)
