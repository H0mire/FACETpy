"""Smoke tests for the DHCT-GAN v2 processor and adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facet.models.dhct_gan_v2 import DHCTGanV2Adapter, DHCTGanV2Correction
from facet.models.dhct_gan_v2.training import (
    DHCTGanV2ContextDataset,
    DHCTGanV2Generator,
    DHCTGanV2Loss,
    PatchGANDiscriminator,
    build_dataset,
    build_loss,
    build_model,
)
from facet.training.dataset import NPZContextArtifactDataset


@pytest.fixture(scope="module")
def epoch_samples() -> int:
    return 512


@pytest.fixture(scope="module")
def context_epochs() -> int:
    return 7


@pytest.fixture(scope="module")
def generator(epoch_samples: int, context_epochs: int) -> DHCTGanV2Generator:
    return build_model(
        epoch_samples=epoch_samples,
        context_epochs=context_epochs,
        base_channels=8,
        depth=3,
    )


def test_build_model_returns_module(generator: DHCTGanV2Generator) -> None:
    assert isinstance(generator, torch.nn.Module)
    assert isinstance(generator, DHCTGanV2Generator)


def test_forward_shape(generator: DHCTGanV2Generator, epoch_samples: int, context_epochs: int) -> None:
    x = torch.randn(2, context_epochs, epoch_samples)
    y = generator(x)
    assert y.shape == (2, 1, epoch_samples)


def test_forward_dict_outputs(
    generator: DHCTGanV2Generator, epoch_samples: int, context_epochs: int
) -> None:
    x = torch.randn(2, context_epochs, epoch_samples)
    out = generator._compute_outputs(x)
    for key in ("artifact", "clean", "fused_clean", "gate"):
        assert out[key].shape == (2, 1, epoch_samples), key
    assert out["noisy_center"].shape == (2, 1, epoch_samples)


def test_backward_updates_gradients(
    generator: DHCTGanV2Generator, epoch_samples: int, context_epochs: int
) -> None:
    x = torch.randn(4, context_epochs, epoch_samples)
    artifact = torch.randn(4, 1, epoch_samples) * 0.1
    clean = torch.randn(4, 1, epoch_samples) * 0.1
    noisy_center = x[:, context_epochs // 2 : context_epochs // 2 + 1, :]
    target = torch.cat([artifact, clean, noisy_center], dim=1)
    loss_fn = build_loss(alpha_consistency=0.5, beta_adv=0.0)

    pred = generator(x)
    loss = loss_fn(pred, target)
    loss.backward()
    assert torch.isfinite(loss)
    grad_count = sum(int(p.grad is not None and torch.any(p.grad != 0)) for p in generator.parameters())
    assert grad_count > 0, "no generator parameters received gradient"


def test_loss_runs_under_no_grad(
    generator: DHCTGanV2Generator, epoch_samples: int, context_epochs: int
) -> None:
    loss_fn = build_loss(beta_adv=0.1)
    x = torch.randn(2, context_epochs, epoch_samples)
    target = torch.cat(
        [
            torch.randn(2, 1, epoch_samples),
            torch.randn(2, 1, epoch_samples),
            x[:, context_epochs // 2 : context_epochs // 2 + 1, :],
        ],
        dim=1,
    )
    with torch.no_grad():
        pred = generator(x)
        value = loss_fn(pred, target)
    assert torch.isfinite(value)


def test_discriminator_shape(epoch_samples: int) -> None:
    disc = PatchGANDiscriminator(in_channels=1, base_channels=8, depth=3)
    out = disc(torch.randn(2, 1, epoch_samples))
    assert out.ndim == 3
    assert out.shape[0] == 2
    assert out.shape[1] == 1


def _write_fixture_npz(path: Path, *, examples: int, channels: int, samples: int, context: int) -> None:
    rng = np.random.default_rng(0)
    noisy_context = rng.standard_normal((examples, context, channels, samples)).astype(np.float32)
    # Center epoch is noisy_context[:, context // 2]
    center = noisy_context[:, context // 2]
    clean = rng.standard_normal((examples, channels, samples)).astype(np.float32) * 0.05
    artifact = center - clean
    noisy_center = center
    np.savez(
        path,
        noisy_context=noisy_context,
        noisy_center=noisy_center,
        clean_center=clean,
        artifact_center=artifact,
        sfreq=np.array([512.0]),
    )


def test_dataset_factory(tmp_path: Path, context_epochs: int) -> None:
    npz = tmp_path / "fixture.npz"
    _write_fixture_npz(npz, examples=4, channels=3, samples=64, context=context_epochs)

    ds = build_dataset(path=str(npz), context_epochs=context_epochs, demean_input=False, demean_target=False)
    assert len(ds) == 4 * 3
    noisy_window, target_stack = ds[0]
    assert noisy_window.shape == (context_epochs, 64)
    assert target_stack.shape == (3, 64)


def test_dataset_clean_center_alignment(tmp_path: Path, context_epochs: int) -> None:
    npz = tmp_path / "fixture.npz"
    _write_fixture_npz(npz, examples=2, channels=2, samples=32, context=context_epochs)
    ds = build_dataset(path=str(npz), context_epochs=context_epochs, demean_input=False, demean_target=False)
    noisy, target = ds[0]
    artifact, clean, noisy_center = target[0], target[1], target[2]
    # By construction in the fixture: noisy_center == clean + artifact (with float noise from demean=False)
    np.testing.assert_allclose(noisy_center, clean + artifact, atol=1e-5)


def test_adapter_predict_with_synthetic_raw(tmp_path: Path, context_epochs: int) -> None:
    model = build_model(epoch_samples=64, context_epochs=context_epochs, base_channels=8, depth=3)
    example = torch.randn(1, context_epochs, 64)
    scripted = torch.jit.trace(model.eval(), example)
    checkpoint = tmp_path / "dhct_gan_v2.ts"
    scripted.save(str(checkpoint))

    import mne
    from facet.core import ProcessingContext

    sfreq = 512.0
    n_channels = 2
    n_samples = 1024
    data = np.random.default_rng(1).standard_normal((n_channels, n_samples)).astype(np.float32)
    info = mne.create_info([f"EEG{i:02d}" for i in range(n_channels)], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    # Need at least context_epochs+1 triggers for v2 inference.
    triggers = np.array([50, 150, 250, 350, 450, 550, 650, 750, 850], dtype=int)
    context = ProcessingContext(raw=raw)
    context = context.with_triggers(triggers)

    adapter = DHCTGanV2Adapter(
        checkpoint_path=str(checkpoint),
        context_epochs=context_epochs,
        epoch_samples=64,
        device="cpu",
        artifact_to_trigger_offset=0.0,
        demean_input=True,
        remove_prediction_mean=True,
    )
    adapter.validate_context(context)
    prediction = adapter.predict(context)
    assert prediction.artifact_data is not None
    assert prediction.artifact_data.shape == data.shape
    assert prediction.metadata["corrected_epochs"] >= 1
    assert prediction.metadata["context_epochs"] == context_epochs


def test_adapter_rejects_insufficient_triggers(tmp_path: Path, context_epochs: int) -> None:
    model = build_model(epoch_samples=32, context_epochs=context_epochs, base_channels=8, depth=2)
    example = torch.randn(1, context_epochs, 32)
    scripted = torch.jit.trace(model.eval(), example)
    checkpoint = tmp_path / "dhct_gan_v2.ts"
    scripted.save(str(checkpoint))

    import mne
    from facet.core import ProcessingContext, ProcessorValidationError

    info = mne.create_info(["EEG01", "EEG02"], 512.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 1024), dtype=np.float32), info, verbose=False)
    context = ProcessingContext(raw=raw).with_triggers(np.array([100, 300, 500], dtype=int))

    adapter = DHCTGanV2Adapter(
        checkpoint_path=str(checkpoint),
        context_epochs=context_epochs,
        epoch_samples=32,
        device="cpu",
        artifact_to_trigger_offset=0.0,
    )
    with pytest.raises(ProcessorValidationError):
        adapter.validate_context(context)
