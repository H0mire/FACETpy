"""Regression tests locking the train/inference demean symmetry per model.

Context (review finding "M2"): the input demean a model applies at inference
*must* match the demean its training dataset applied, otherwise the network is
served data on a different scale than it saw during training. The three models
below each use a *different* (but internally consistent) convention:

* ``demucs`` flattens the 7-epoch context into one waveform and subtracts a
  single **global** mean (its training ``FlatContextArtifactDataset`` flattens
  first, then demeans ``axis=-1``).
* ``dpae`` and ``dhct_gan`` operate on a **single** epoch, where ``x.mean()``
  is by definition the per-epoch demean.

These tests feed identical epoch data through (a) the real training dataset's
``__getitem__`` and (b) the real inference ``predict()`` — capturing the exact
array handed to the model via a recording stub — and assert the demeaned model
input matches. A future change that flips a processor between global and
per-epoch demean (the failure mode M2 warned about) will fail here.

The epochs are built at native length == ``epoch_samples`` so the polyphase
resampler is an identity and the comparison isolates the demean step.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest


def _make_context(data: np.ndarray, triggers: list[int], sfreq: float = 100.0):
    from facet.core import ProcessingContext, ProcessingMetadata

    info = mne.create_info(
        ch_names=[f"EEG{i + 1:03d}" for i in range(data.shape[0])],
        sfreq=sfreq,
        ch_types=["eeg"] * data.shape[0],
    )
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    metadata = ProcessingMetadata(
        triggers=np.asarray(triggers, dtype=int),
        artifact_to_trigger_offset=0.0,
    )
    return ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=metadata)


def _recording_stub(torch):
    """A torch Module that records every input it receives and returns zeros."""

    class _Stub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: list[np.ndarray] = []

        def forward(self, x):  # noqa: D401 - simple passthrough recorder
            self.inputs.append(x.detach().cpu().numpy().copy())
            return torch.zeros_like(x)

    return _Stub().eval()


@pytest.mark.unit
def test_demucs_inference_demean_matches_flat_context_dataset(tmp_path):
    torch = pytest.importorskip("torch")
    from facet.models.demucs.processor import DemucsAdapter
    from facet.models.demucs.training import FlatContextArtifactDataset

    samples = 16
    n_epochs = 7
    rng = np.random.default_rng(0)
    # Per-epoch DC offsets make a GLOBAL demean differ observably from a
    # per-epoch demean, so the test can distinguish the two conventions.
    epochs = rng.standard_normal((n_epochs, samples))
    epochs += np.arange(1, n_epochs + 1)[:, None] * 5.0
    data = epochs.reshape(1, n_epochs * samples)
    triggers = list(range(0, n_epochs * samples + 1, samples))  # n_epochs + 1
    context = _make_context(data, triggers)

    adapter = DemucsAdapter(
        tmp_path / "unused.ts",
        context_epochs=n_epochs,
        epoch_samples=samples,
        demean_input=True,
        remove_prediction_mean=False,
        eeg_only=False,
    )
    stub = _recording_stub(torch)
    adapter._model = stub
    adapter._torch = torch
    adapter.predict(context)

    assert len(stub.inputs) == 1, "expected a single channel/center forward pass"
    captured = stub.inputs[0].reshape(-1)  # flattened 7-epoch waveform fed to model

    # Training side: identical epochs through the real dataset.
    npz = tmp_path / "bundle.npz"
    noisy_context = epochs.reshape(1, n_epochs, 1, samples).astype(np.float32)
    np.savez(
        npz,
        noisy_context=noisy_context,
        artifact_context=noisy_context * 0.5,
        sfreq=np.array([100.0]),
    )
    dataset = FlatContextArtifactDataset(
        npz, context_epochs=n_epochs, demean_input=True, demean_target=False
    )
    train_input, _ = dataset[0]

    np.testing.assert_allclose(captured, train_input.reshape(-1), atol=1e-5)

    # Guard: this is genuinely a global demean, not a per-epoch one.
    per_epoch = (epochs - epochs.mean(axis=-1, keepdims=True)).reshape(-1)
    assert not np.allclose(captured, per_epoch, atol=1e-3)


@pytest.mark.unit
def test_dpae_inference_demean_matches_channelwise_dataset(tmp_path):
    torch = pytest.importorskip("torch")
    from facet.models.dpae.processor import DualPathwayAutoencoderAdapter
    from facet.models.dpae.training import ChannelWiseArtifactDataset
    from facet.training.dataset import NPZContextArtifactDataset

    samples = 16
    n_epochs = 7
    rng = np.random.default_rng(1)
    epoch = rng.standard_normal(samples) + 7.0  # single epoch with DC offset
    data_epochs = rng.standard_normal((n_epochs, samples))
    data_epochs[0] = epoch
    data = data_epochs.reshape(1, n_epochs * samples)
    triggers = list(range(0, n_epochs * samples + 1, samples))
    context = _make_context(data, triggers)

    adapter = DualPathwayAutoencoderAdapter(
        tmp_path / "unused.ts",
        epoch_samples=samples,
        demean_input=True,
        remove_prediction_mean=False,
        eeg_only=False,
    )
    stub = _recording_stub(torch)
    adapter._model = stub
    adapter._torch = torch
    adapter.predict(context)

    captured = stub.inputs[0].reshape(-1)  # first epoch fed to the model

    # Training side: place the same epoch at the context center the dataset reads.
    npz = tmp_path / "bundle.npz"
    center = n_epochs // 2
    noisy_context = np.zeros((1, n_epochs, 1, samples), dtype=np.float32)
    noisy_context[0, center, 0, :] = epoch
    artifact_center = (noisy_context[:, center] * 0.5).astype(np.float32)
    np.savez(npz, noisy_context=noisy_context, artifact_center=artifact_center, sfreq=np.array([100.0]))
    base = NPZContextArtifactDataset(npz, demean_input=False, demean_target=False)
    dataset = ChannelWiseArtifactDataset(base, demean_input=True, demean_target=False)
    train_input, _ = dataset[0]

    np.testing.assert_allclose(captured, train_input.reshape(-1), atol=1e-5)


@pytest.mark.unit
def test_dhct_gan_inference_demean_matches_single_epoch_dataset(tmp_path):
    torch = pytest.importorskip("torch")
    from facet.models.dhct_gan.processor import DHCTGanAdapter
    from facet.models.dhct_gan.training import DHCTGanArtifactDataset

    samples = 16
    n_epochs = 7
    rng = np.random.default_rng(2)
    epoch = rng.standard_normal(samples) + 4.0
    data_epochs = rng.standard_normal((n_epochs, samples))
    data_epochs[0] = epoch
    data = data_epochs.reshape(1, n_epochs * samples)
    triggers = list(range(0, n_epochs * samples + 1, samples))
    context = _make_context(data, triggers)

    adapter = DHCTGanAdapter(
        tmp_path / "unused.ts",
        epoch_samples=samples,
        demean_input=True,
        remove_prediction_mean=False,
        eeg_only=False,
    )
    stub = _recording_stub(torch)
    adapter._model = stub
    adapter._torch = torch
    adapter.predict(context)

    captured = stub.inputs[0].reshape(-1)  # first epoch fed to the model

    npz = tmp_path / "bundle.npz"
    noisy_center = np.zeros((1, 1, samples), dtype=np.float32)
    noisy_center[0, 0, :] = epoch
    np.savez(
        npz,
        noisy_center=noisy_center,
        clean_center=np.zeros_like(noisy_center),
        artifact_center=noisy_center * 0.5,
        sfreq=np.array([100.0]),
    )
    dataset = DHCTGanArtifactDataset(npz, demean=True)
    train_input, _ = dataset[0]

    np.testing.assert_allclose(captured, train_input.reshape(-1), atol=1e-5)
