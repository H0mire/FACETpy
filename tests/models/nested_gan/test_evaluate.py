from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.mark.unit
def test_evaluate_writes_standard_run_files(tmp_path):
    torch = pytest.importorskip("torch")
    from facet.models.nested_gan.evaluate import main as evaluate_main

    class Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device)

    checkpoint = tmp_path / "zero_nested_gan.ts"
    torch.jit.trace(Zero(), torch.zeros(1, 7, 1, 32)).save(str(checkpoint))

    dataset_path = tmp_path / "bundle.npz"
    rng = np.random.default_rng(0)
    n_examples, n_channels, n_samples = 3, 2, 32
    artifact = rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32)
    clean = 0.1 * rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32)
    noisy = clean + artifact
    context = np.tile(noisy[:, None, :, :], (1, 7, 1, 1)).astype(np.float32)
    np.savez(
        dataset_path,
        noisy_context=context,
        noisy_center=noisy,
        artifact_center=artifact,
        clean_center=clean,
        sfreq=np.asarray([5000.0]),
        ch_names=np.asarray(["C3", "C4"]),
    )

    output_root = tmp_path / "output_root"
    docs_root = tmp_path / "docs_root"
    rc = evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(dataset_path),
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--run-id",
            "smoke_test",
            "--output-root",
            str(output_root),
            "--docs-root",
            str(docs_root),
        ]
    )
    assert rc == 0

    run_dir = output_root / "nested_gan" / "smoke_test"
    assert (run_dir / "evaluation_manifest.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "evaluation_summary.md").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    flat = metrics["flat_metrics"]
    # Zero predictor → corrected = noisy; before/after metrics must match exactly.
    assert flat["synthetic.clean_snr_db_before"] == pytest.approx(flat["synthetic.clean_snr_db_after"])
    # Artifact correlation between zero prediction and ground-truth artifact is undefined/zero.
    assert flat["synthetic.artifact_correlation"] == pytest.approx(0.0)
