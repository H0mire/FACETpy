from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_builder_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "build_niazy_proof_fit_context_dataset.py"
    spec = importlib.util.spec_from_file_location("build_niazy_proof_fit_context_dataset", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_context_dataset_uses_corrected_plus_artifact_as_noisy():
    builder = _load_builder_module()
    artifact = np.ones((2, 80), dtype=np.float32) * 3.0
    corrected = np.ones((2, 80), dtype=np.float32) * 2.0
    bundle = {
        "artifact": artifact,
        "corrected": corrected,
        "triggers": np.arange(0, 80, 8, dtype=np.int64),
        "sfreq": np.asarray([1000.0], dtype=np.float64),
        "artifact_to_trigger_offset": np.asarray([0.0], dtype=np.float64),
        "ch_names": np.asarray(["C3", "C4"], dtype=object),
        "_source_path": np.asarray("toy.npz", dtype=object),
    }

    dataset = builder._build_context_dataset(
        bundle,
        context_epochs=3,
        target_epoch_samples=8,
    )

    assert dataset["noisy_context"].shape == (7, 3, 2, 8)
    np.testing.assert_allclose(dataset["clean_context"], 2.0)
    np.testing.assert_allclose(dataset["artifact_context"], 3.0)
    np.testing.assert_allclose(dataset["noisy_context"], 5.0)
    np.testing.assert_allclose(dataset["artifact_center"], 3.0)
