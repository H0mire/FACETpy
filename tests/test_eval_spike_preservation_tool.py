"""Tests for the spike-preservation evaluation tool's data handling.

The tool decides three things that every downstream number depends on: which
window is cut, whether the template is cut with it, and what the cascade arm is
subtracted from. Those are exactly the places where a silent mistake produces a
plausible-looking result, so they are pinned here against a tiny synthetic
dataset with known content.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TOOL = Path("tools/evaluation/eval_run6_spike_preservation.py")

_SPEC = importlib.util.spec_from_file_location("eval_run6", TOOL)
evaltool = importlib.util.module_from_spec(_SPEC)
sys.modules["eval_run6"] = evaltool
_SPEC.loader.exec_module(evaltool)


CORE, GUARD, EPOCHS, CHANNELS = 64, 8, 3, 2
TOTAL = CORE + 2 * GUARD


def _dataset(tmp_path: Path, n=12, n_events=4) -> Path:
    """Synthetic Weg-A-shaped NPZ with a known template and known spike events.

    Electrode replicates of one event share a ``center_epoch_index``, which is
    what the clustering in the paired comparison keys on.
    """
    rng = np.random.default_rng(0)
    clean_ctx = rng.standard_normal((n, EPOCHS, CHANNELS, TOTAL)).astype(np.float32) * 1e-6
    template_ctx = np.tile(
        np.sin(np.linspace(0, 8 * np.pi, TOTAL, dtype=np.float32))[None, None, None, :],
        (n, EPOCHS, CHANNELS, 1),
    ) * 1e-4
    residual_ctx = rng.standard_normal((n, EPOCHS, CHANNELS, TOTAL)).astype(np.float32) * 1e-5
    artifact_ctx = template_ctx + residual_ctx

    clean_c = clean_ctx[:, EPOCHS // 2, 0:1, :].copy()
    template_c = template_ctx[:, EPOCHS // 2, 0:1, :].copy()
    artifact_c = artifact_ctx[:, EPOCHS // 2, 0:1, :].copy()

    spikes = np.zeros((n, 1, TOTAL), dtype=np.float32)
    per_event = n // (2 * n_events)                      # half the rows are val
    epoch_index = np.zeros(n, dtype=np.int64)
    split = np.zeros(n, dtype=np.int64)
    row = 0
    for event in range(2 * n_events):
        for _ in range(max(1, per_event)):
            if row >= n:
                break
            epoch_index[row] = 100 + event
            split[row] = 1 if event >= n_events else 0
            spikes[row, 0, GUARD + CORE // 2 - 2: GUARD + CORE // 2 + 3] = 1.0
            clean_c[row, 0, GUARD + CORE // 2 - 2: GUARD + CORE // 2 + 3] += 5e-5
            row += 1

    path = tmp_path / "ds.npz"
    np.savez(
        path,
        clean_context=clean_ctx, artifact_context=artifact_ctx,
        artifact_context_template=template_ctx,
        clean_center=clean_c, artifact_center=artifact_c, artifact_center_template=template_c,
        spike_labels=spikes,
        neighbor_channel_indices=np.tile(np.arange(CHANNELS)[None, :], (n, 1)).astype(np.int64),
        target_channel_index=np.zeros(n, dtype=np.int64),
        center_epoch_index=epoch_index,
        example_split=split,
        core_samples=np.array([CORE]), guard_samples=np.array([GUARD]),
        context_epochs=np.array([EPOCHS]), k_neighbors=np.array([CHANNELS - 1]),
        sfreq=np.array([1024.0]), n_examples=np.array([n]),
    )
    return path


#: A model factory the tool imports from a subprocess. It is written into the
#: temporary directory rather than imported from the test module, because the
#: tool runs as its own process and ``tests`` is not an installed package.
ZERO_FACTORY = '\n'.join([
    'import torch',
    '',
    '',
    'class ZeroArtifactModel(torch.nn.Module):',
    '    # Predicts exactly zero, so the model arm equals its own input.',
    '    def __init__(self, input_shape=None, **_):',
    '        super().__init__()',
    '        self.core = int(input_shape[2]) if input_shape else 64',
    '        self.unused = torch.nn.Parameter(torch.zeros(1))',
    '',
    '    def forward(self, x):',
    '        return torch.zeros((x.shape[0], 1, self.core), dtype=x.dtype, device=x.device)',
    '',
    '',
    'def build_model(input_shape=None, **kwargs):',
    '    return ZeroArtifactModel(input_shape=input_shape, **kwargs)',
    '',
])


class _Zero(torch.nn.Module):
    """In-process twin of the factory model, used to write the checkpoint."""

    def __init__(self, input_shape=None, **_):
        super().__init__()
        self.core = int(input_shape[2]) if input_shape else CORE
        self.unused = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros((x.shape[0], 1, self.core), dtype=x.dtype, device=x.device)


def _checkpoint(tmp_path: Path) -> Path:
    model = _Zero(input_shape=(EPOCHS, CHANNELS, CORE))
    path = tmp_path / "zero.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    (tmp_path / "zero_factory.py").write_text(ZERO_FACTORY, encoding="utf-8")
    return path


def _run(tmp_path: Path, dataset: Path, ckpt: Path, extra: list[str]) -> dict:
    out = tmp_path / f"out{abs(hash(tuple(extra))) % 10000}"
    cmd = [
        sys.executable, str(TOOL), "--dataset", str(dataset), "--checkpoint", str(ckpt),
        "--model-factory", "zero_factory:build_model",
        "--model-kwargs", "{}", "--device", "cpu", "--output-dir", str(out), *extra,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), "src", env.get("PYTHONPATH", "")])
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd(), env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads((out / "run6_spike_preservation.json").read_text())


@pytest.mark.unit
def test_zero_prediction_leaves_the_model_arm_equal_to_the_raw_signal(tmp_path):
    """A model predicting nothing must score exactly like 'no correction'."""
    ds, ck = _dataset(tmp_path), _checkpoint(tmp_path)
    meta = _run(tmp_path, ds, ck, [])
    # corrected = noisy - 0 = noisy, so the error is the artifact itself.
    with np.load(ds) as b:
        sl = slice(GUARD, GUARD + CORE)
        val = np.flatnonzero(b["example_split"] == 1)
        artifact = b["artifact_center"][val][:, 0, sl].astype(np.float64)
    expected = float(np.sqrt(np.mean(artifact ** 2))) * 1e6
    assert meta["results"]["model"]["overall_rmse_uv"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.unit
def test_residual_mode_subtracts_from_the_farm_corrected_signal(tmp_path):
    """In residual mode a zero prediction must equal the ideal-AAS arm exactly."""
    ds, ck = _dataset(tmp_path), _checkpoint(tmp_path)
    meta = _run(tmp_path, ds, ck, ["--residual-mode"])
    assert meta["residual_mode"] is True
    model = meta["results"]["model"]["overall_rmse_uv"]
    aas = meta["results"]["aas_ideal"]["overall_rmse_uv"]
    assert model == pytest.approx(aas, rel=1e-9), (
        "residual mode must subtract the prediction from 'noisy - template', not from 'noisy'"
    )


@pytest.mark.unit
def test_null_arm_equals_the_rms_of_the_true_clean(tmp_path):
    """The null arm is the signal-deletion baseline; its error is RMS(clean)."""
    ds, ck = _dataset(tmp_path), _checkpoint(tmp_path)
    meta = _run(tmp_path, ds, ck, [])
    with np.load(ds) as b:
        sl = slice(GUARD, GUARD + CORE)
        val = np.flatnonzero(b["example_split"] == 1)
        clean = b["clean_center"][val][:, 0, sl].astype(np.float64)
    assert meta["results"]["null_output"]["overall_rmse_uv"] == pytest.approx(
        float(np.sqrt(np.mean(clean ** 2))) * 1e6, rel=1e-9
    )


@pytest.mark.unit
def test_window_shift_moves_signal_and_template_together(tmp_path):
    """A common shift must leave the ideal-AAS arm unchanged in structure.

    Everything moves together, so the AAS arm still removes exactly the template
    that belongs to its window. The value may differ because a different stretch
    of signal is scored, but the *template alignment* must be preserved — which
    the misalignment test below deliberately breaks.
    """
    ds, ck = _dataset(tmp_path), _checkpoint(tmp_path)
    base = _run(tmp_path, ds, ck, [])
    shifted = _run(tmp_path, ds, ck, ["--window-shift", "4"])
    assert shifted["window_shift_samples"] == 4
    assert shifted["trigger_misalign_samples"] == 0
    aligned_error = shifted["results"]["aas_ideal"]["overall_rmse_uv"]
    assert aligned_error < 5 * base["results"]["aas_ideal"]["overall_rmse_uv"]


@pytest.mark.unit
def test_trigger_misalignment_degrades_the_template_arm(tmp_path):
    """Shifting only the template must make the AAS arm markedly worse.

    This is the regression guard for the position sensitivity finding: with a
    periodic template, a few samples of misalignment turn a near-perfect
    subtraction into an error larger than the template itself.
    """
    ds, ck = _dataset(tmp_path), _checkpoint(tmp_path)
    base = _run(tmp_path, ds, ck, [])
    off = _run(tmp_path, ds, ck, ["--trigger-misalign", "4"])
    assert off["trigger_misalign_samples"] == 4
    assert off["results"]["aas_ideal"]["overall_rmse_uv"] > (
        3.0 * base["results"]["aas_ideal"]["overall_rmse_uv"]
    )


@pytest.mark.unit
def test_per_example_csv_carries_the_event_id_and_the_bulk_table_exists(tmp_path):
    """Both output tables and the clustering key must be written."""
    ds, ck = _dataset(tmp_path), _checkpoint(tmp_path)
    meta = _run(tmp_path, ds, ck, ["--batch-size", "4"])
    out = tmp_path / f"out{abs(hash(('--batch-size', '4'))) % 10000}"
    header = (out / "run6_spike_preservation_per_example.csv").read_text().splitlines()[0]
    assert "spike_event_id" in header and "target_channel" in header
    bulk = (out / "run6_bulk_per_example.csv").read_text().splitlines()
    assert "epoch_id" in bulk[0] and "clean_snr_db" in bulk[0]
    assert meta["per_example"]["n_spike_events"] >= 1
    assert meta["per_example"]["n_bulk_epochs"] >= 1
