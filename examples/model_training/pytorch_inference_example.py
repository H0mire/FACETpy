"""
PyTorch inference adapter — minimal end-to-end example with a TorchScript model.

This example:

1. creates a tiny synthetic EEG recording,
2. exports a small TorchScript artifact model to a temporary checkpoint,
3. runs ``DeepLearningCorrection("pytorch_inference")``,
4. prints the correction metadata stored in the FACETpy context.

Requires the optional PyTorch extra:

    pip install facetpy[pytorch]

or with Poetry:

    poetry install -E pytorch
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mne
import numpy as np
import torch

from facet import (
    DeepLearningArchitecture,
    DeepLearningCorrection,
    DeepLearningExecutionGranularity,
    DeepLearningOutputType,
    DeepLearningRuntime,
    Pipeline,
    ProcessingContext,
)


class SimpleArtifactModel(torch.nn.Module):
    """Toy artifact estimator that predicts a fixed fraction of the input."""

    def forward(self, x):
        return x * 0.2


def build_context() -> ProcessingContext:
    """Create a small synthetic EEG recording."""
    sfreq = 250.0
    n_samples = 1000
    times = np.arange(n_samples) / sfreq

    data = np.vstack(
        [
            np.sin(2 * np.pi * 8.0 * times),
            np.sin(2 * np.pi * 12.0 * times + 0.3),
            np.sin(2 * np.pi * 18.0 * times + 0.7),
            np.sin(2 * np.pi * 25.0 * times + 1.1),
        ]
    ).astype(np.float32)

    # Add a simple repeating artifact component so the correction effect is visible.
    artifact = np.zeros_like(data)
    artifact[:, 200:260] += 0.5
    artifact[:, 500:560] -= 0.4
    raw = mne.io.RawArray(
        data + artifact,
        mne.create_info(
            ch_names=["EEG001", "EEG002", "EEG003", "EEG004"],
            sfreq=sfreq,
            ch_types=["eeg", "eeg", "eeg", "eeg"],
        ),
        verbose=False,
    )
    return ProcessingContext(raw=raw, raw_original=raw.copy())


def export_torchscript_checkpoint(path: Path) -> Path:
    """Export the toy artifact model as TorchScript."""
    model = SimpleArtifactModel().eval()
    example_input = torch.randn(1, 4, 1000)
    scripted = torch.jit.trace(model, example_input)
    scripted.save(str(path))
    return path


def main() -> None:
    context = build_context()

    with tempfile.TemporaryDirectory(prefix="facetpy-pytorch-example-") as tmp_dir:
        checkpoint_path = export_torchscript_checkpoint(Path(tmp_dir) / "simple_artifact_model.ts")

        processor = DeepLearningCorrection(
            "pytorch_inference",
            model_kwargs={
                "checkpoint_path": str(checkpoint_path),
                "spec_overrides": {
                    "name": "ExampleTorchScriptArtifactModel",
                    "architecture": DeepLearningArchitecture.AUTOENCODER,
                    "runtime": DeepLearningRuntime.PYTORCH,
                    "output_type": DeepLearningOutputType.ARTIFACT,
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                    "device_preference": "cpu",
                },
            },
        )

        result = Pipeline([processor], name="PyTorch inference example").run(
            initial_context=context,
            show_progress=False,
        )

    run = result.context.metadata.custom["deep_learning_runs"][0]
    corrected = result.context.get_raw().get_data(copy=False)
    estimated_noise = result.context.get_estimated_noise()

    print("Stored run metadata:")
    print(f"  model: {run['model']}")
    print(f"  runtime: {run['runtime']}")
    print(f"  device: {run['prediction_metadata']['device']}")
    print(f"  checkpoint_format: {run['prediction_metadata']['checkpoint_format']}")
    print(f"  checkpoint_load_mode: {run['prediction_metadata']['checkpoint_load_mode']}")
    print()
    print(f"Corrected data shape: {corrected.shape}")
    print(f"Estimated noise shape: {estimated_noise.shape}")
    print(f"Mean estimated artifact amplitude: {estimated_noise.mean():.4f}")


if __name__ == "__main__":
    main()
