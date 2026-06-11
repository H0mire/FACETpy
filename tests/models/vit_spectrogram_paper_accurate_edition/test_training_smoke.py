"""CPU smoke test for the paper-accurate MAE-ViT spectrogram inpainter.

Every architecture dimension is shrunk so forward+backward finishes in a few
seconds on CPU. Asserts:

1. the inference (eval) forward returns ``(B, 1, epoch_samples)``;
2. the MAE training-mode forward returns the masked-patch dict and the
   masked-patch magnitude loss STRICTLY decreases over a few AdamW steps;
3. a ``torch.jit.trace`` round-trip of the inference forward saves+reloads and
   reproduces the output within ``atol 1e-5``;
4. the dataset factory round-trips through a tiny in-memory NPZ bundle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facet.models.vit_spectrogram_paper_accurate_edition.training import (
    ChannelWiseSpectrogramDataset,
    MaskedPatchMagnitudeLoss,
    ViTSpectrogramMAEInpainter,
    build_dataset,
    build_loss,
    build_model,
)

# Tiny smoke geometry (see ANALYSIS smoke_plan).
CONTEXT_EPOCHS = 7
EPOCH_SAMPLES = 64
N_FFT = 16
HOP = 8
FREQ_BINS = 8
# time_frames must cover the full context_epochs * epoch_samples = 448-sample
# signal (~57 STFT frames at hop 8) so the iSTFT reconstruction actually
# contains the center epoch; 56 is divisible by patch_time and stays cheap.
TIME_FRAMES = 56
PATCH_FREQ = 4
PATCH_TIME = 4
BATCH = 4

_MODEL_KW = dict(
    input_shape=(CONTEXT_EPOCHS, 1, EPOCH_SAMPLES),
    n_fft=N_FFT,
    hop_length=HOP,
    freq_bins=FREQ_BINS,
    time_frames=TIME_FRAMES,
    patch_freq=PATCH_FREQ,
    patch_time=PATCH_TIME,
    embed_dim=16,
    depth=2,
    n_heads=2,
    mlp_ratio=2.0,
    dropout=0.0,
    decoder_embed_dim=8,
    decoder_depth=1,
    decoder_heads=2,
    decoder_mlp_ratio=2.0,
    mask_margin_patches=1,
)

_LOSS_KW = dict(
    name="mse",
    input_shape=(CONTEXT_EPOCHS, 1, EPOCH_SAMPLES),
    n_fft=N_FFT,
    hop_length=HOP,
    freq_bins=FREQ_BINS,
    time_frames=TIME_FRAMES,
    patch_freq=PATCH_FREQ,
    patch_time=PATCH_TIME,
    mask_margin_patches=1,
    normalize_target=True,
)


@pytest.mark.unit
def test_inference_forward_shape():
    torch = pytest.importorskip("torch")
    model = build_model(**_MODEL_KW).eval()
    x = torch.randn(BATCH, CONTEXT_EPOCHS, 1, EPOCH_SAMPLES)
    with torch.no_grad():
        out = model(x)
    assert tuple(out.shape) == (BATCH, 1, EPOCH_SAMPLES)
    # The iSTFT reconstruction must actually cover the center epoch (guards
    # against a time_frames crop that drops the center window -> all-zeros).
    assert torch.isfinite(out).all()
    assert float(out.std()) > 0.0


@pytest.mark.unit
def test_training_forward_is_asymmetric_mae_dict():
    torch = pytest.importorskip("torch")
    model = build_model(**_MODEL_KW).train()
    # Encoder must see ONLY the visible (non-masked) tokens (MAE Design A).
    assert model.n_masked_patches >= 1
    assert model.n_visible_patches >= 1
    assert model.n_visible_patches + model.n_masked_patches == model.n_patches
    x = torch.randn(BATCH, CONTEXT_EPOCHS, 1, EPOCH_SAMPLES) * 1e-3
    out = model(x)
    assert isinstance(out, dict)
    assert out["pred_masked_patches"].shape == (BATCH, model.n_masked_patches, model.patch_pixels)


@pytest.mark.unit
def test_masked_patch_loss_decreases():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = build_model(**_MODEL_KW).train()
    loss_fn = build_loss(**_LOSS_KW)
    assert isinstance(loss_fn, MaskedPatchMagnitudeLoss)

    rng = np.random.default_rng(0)
    noisy = torch.as_tensor(
        rng.standard_normal((BATCH, CONTEXT_EPOCHS, 1, EPOCH_SAMPLES)).astype(np.float32) * 1e-3
    )
    target = torch.as_tensor(rng.standard_normal((BATCH, 1, EPOCH_SAMPLES)).astype(np.float32) * 1e-3)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)

    initial = float(loss_fn(model(noisy), target).detach())
    final = initial
    for _ in range(6):
        opt.zero_grad()
        loss = loss_fn(model(noisy), target)
        loss.backward()
        opt.step()
        final = float(loss.detach())
    assert final < initial, f"loss did not decrease: initial={initial}, final={final}"


@pytest.mark.unit
def test_torchscript_trace_round_trip(tmp_path: Path):
    torch = pytest.importorskip("torch")
    model = build_model(**_MODEL_KW).eval()
    example = torch.randn(1, CONTEXT_EPOCHS, 1, EPOCH_SAMPLES)
    traced = torch.jit.trace(model, example)
    artifact_path = tmp_path / "model.ts"
    traced.save(str(artifact_path))
    assert artifact_path.exists()

    reloaded = torch.jit.load(str(artifact_path))
    reloaded.eval()
    with torch.no_grad():
        a = model(example)
        b = reloaded(example)
    assert tuple(a.shape) == tuple(b.shape) == (1, 1, EPOCH_SAMPLES)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.unit
def test_dataset_factory_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    n_examples, n_channels, samples = 4, 2, EPOCH_SAMPLES
    bundle = {
        "noisy_context": rng.standard_normal((n_examples, CONTEXT_EPOCHS, n_channels, samples)).astype(np.float32),
        "clean_center": rng.standard_normal((n_examples, n_channels, samples)).astype(np.float32),
        "sfreq": np.asarray([2048.0]),
    }
    path = tmp_path / "tiny.npz"
    np.savez_compressed(path, **bundle)
    dataset = build_dataset(path=str(path), context_epochs=CONTEXT_EPOCHS, demean_input=False, demean_target=False)
    assert isinstance(dataset, ChannelWiseSpectrogramDataset)
    assert len(dataset) == n_examples * n_channels
    noisy, target = dataset[0]
    assert noisy.shape == (CONTEXT_EPOCHS, 1, samples)
    assert target.shape == (1, samples)
    assert dataset.input_shape == (CONTEXT_EPOCHS, 1, samples)
    assert dataset.target_shape == (1, samples)
    assert dataset.target_type == "clean"
    assert dataset.trigger_aligned is True
    train_ds, val_ds = dataset.train_val_split(val_ratio=0.25, seed=1)
    assert len(train_ds) + len(val_ds) == len(dataset)
