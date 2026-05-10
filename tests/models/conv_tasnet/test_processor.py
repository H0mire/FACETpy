from __future__ import annotations

import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.models.conv_tasnet import ConvTasNetCorrection
from facet.models.conv_tasnet.training import (
    ChannelWiseSourceSeparationDataset,
    ConvTasNetSeparator,
    build_loss,
    build_model,
)


def _tiny_separator(torch, *, samples: int = 32) -> ConvTasNetSeparator:
    return ConvTasNetSeparator(
        n_sources=2,
        encoder_filters=8,
        encoder_kernel=4,
        bottleneck_channels=4,
        hidden_channels=8,
        block_kernel=3,
        n_blocks=3,
        n_repeats=1,
        mask_activation="sigmoid",
    )


def test_build_model_returns_torch_module():
    torch = pytest.importorskip("torch")
    model = build_model(
        encoder_filters=8,
        encoder_kernel=4,
        bottleneck_channels=4,
        hidden_channels=8,
        block_kernel=3,
        n_blocks=2,
        n_repeats=1,
    )
    assert isinstance(model, torch.nn.Module)


def test_forward_pass_produces_two_source_output_shape():
    torch = pytest.importorskip("torch")
    model = _tiny_separator(torch, samples=32)
    mixture = torch.randn(3, 1, 32)
    out = model(mixture)
    assert tuple(out.shape) == (3, 2, 32)


def test_one_batch_backward_pass_updates_parameters():
    torch = pytest.importorskip("torch")
    model = _tiny_separator(torch, samples=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    initial = [p.detach().clone() for p in model.parameters()]
    mixture = torch.randn(4, 1, 32)
    target = torch.randn(4, 2, 32)

    optimizer.zero_grad()
    pred = model(mixture)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    grad_norms = [p.grad.detach().abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert any(norm > 0.0 for norm in grad_norms), "expected at least one non-zero gradient"

    optimizer.step()
    moved = sum(
        1
        for before, p in zip(initial, model.parameters())
        if not torch.allclose(before, p.detach())
    )
    assert moved > 0


def test_loss_factory_supports_named_variants():
    torch = pytest.importorskip("torch")
    assert isinstance(build_loss("mse"), torch.nn.MSELoss)
    assert isinstance(build_loss("l1"), torch.nn.L1Loss)
    weighted = build_loss("weighted_mse", clean_weight=0.3, artifact_weight=2.0)
    assert weighted(torch.zeros(2, 2, 8), torch.ones(2, 2, 8)).item() == pytest.approx(2.3)
    si_sdr = build_loss("si_sdr_neg")
    pred = torch.randn(2, 2, 32)
    out = si_sdr(pred, pred.clone())
    assert torch.isfinite(out)


def test_channel_wise_source_separation_dataset(tmp_path):
    rng = np.random.default_rng(0)
    n_examples, n_channels, n_samples = 5, 3, 16
    bundle = {
        "noisy_center": rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32),
        "clean_center": rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32),
        "artifact_center": rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32),
        "sfreq": np.asarray([5000.0], dtype=np.float64),
    }
    npz_path = tmp_path / "centers.npz"
    np.savez_compressed(npz_path, **bundle)

    dataset = ChannelWiseSourceSeparationDataset(
        path=npz_path,
        demean_input=False,
        demean_target=False,
    )
    assert len(dataset) == n_examples * n_channels
    mixture, sources = dataset[7]
    assert mixture.shape == (1, n_samples)
    assert sources.shape == (2, n_samples)
    np.testing.assert_allclose(sources[0], bundle["clean_center"][7 // n_channels, 7 % n_channels])
    np.testing.assert_allclose(sources[1], bundle["artifact_center"][7 // n_channels, 7 % n_channels])


def test_conv_tasnet_correction_subtracts_predicted_artifact(tmp_path):
    torch = pytest.importorskip("torch")
    mne = pytest.importorskip("mne")

    class _ConstantArtifactSeparator(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            clean = torch.zeros_like(x)
            artifact = torch.full_like(x, 0.25)
            return torch.cat([clean, artifact], dim=1)

    chunk_size = 16
    checkpoint = tmp_path / "constant_conv_tasnet.ts"
    scripted = torch.jit.trace(_ConstantArtifactSeparator(), torch.zeros(1, 1, chunk_size))
    scripted.save(str(checkpoint))

    n_samples = chunk_size * 4
    data = np.ones((2, n_samples), dtype=np.float64)
    info = mne.create_info(["C3", "C4"], sfreq=1000.0, ch_types="eeg")
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    context = ProcessingContext(raw=raw, metadata=ProcessingMetadata())

    result = context | ConvTasNetCorrection(
        checkpoint_path=checkpoint,
        chunk_size_samples=chunk_size,
        chunk_overlap_samples=0,
        demean_input=False,
        remove_prediction_mean=False,
    )

    expected_artifact = np.full_like(data, 0.25)
    np.testing.assert_allclose(result.get_estimated_noise(), expected_artifact)
    np.testing.assert_allclose(result.get_raw()._data, data - expected_artifact)
    runs = result.metadata.custom["deep_learning_runs"]
    assert runs[-1]["model"] == "ConvTasNetAdapter"
    chunk_summaries = runs[-1]["prediction_metadata"]["chunks"]
    assert chunk_summaries, "expected at least one chunk summary"
    assert chunk_summaries[0]["prediction_metadata"]["chunk_size_samples"] == chunk_size
