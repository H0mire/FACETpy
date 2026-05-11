from __future__ import annotations

import numpy as np
import pytest

from facet.models.sepformer.training import build_loss, build_model


def test_sepformer_si_snr_loss_negative_or_zero():
    torch = pytest.importorskip("torch")
    loss = build_loss("si_snr")
    target = torch.randn(2, 1, 64)
    perfect = target.clone()
    # SI-SNR of a perfect prediction is +infinity; negative-SI-SNR loss therefore very negative.
    assert float(loss(perfect, target)) < -50.0


def test_sepformer_si_snr_mse_loss_finite_on_random_pair():
    torch = pytest.importorskip("torch")
    loss = build_loss("si_snr_mse", mse_weight=0.1)
    prediction = torch.randn(2, 1, 64)
    target = torch.randn(2, 1, 64)
    value = float(loss(prediction, target))
    assert np.isfinite(value)


def test_sepformer_torchscript_export_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    model = build_model(
        epoch_samples=64,
        context_epochs=7,
        encoder_channels=16,
        encoder_kernel=8,
        encoder_stride=4,
        chunk_size=16,
        n_blocks=1,
        intra_layers=1,
        inter_layers=1,
        intra_heads=4,
        inter_heads=4,
        d_ffn=32,
        dropout=0.0,
    )
    model.eval()
    example = torch.zeros(1, 7, 1, 64)
    scripted = torch.jit.trace(model, example)
    path = tmp_path / "sepformer_smoke.ts"
    scripted.save(str(path))

    loaded = torch.jit.load(str(path), map_location="cpu")
    out = loaded(torch.randn(2, 7, 1, 64))
    assert tuple(out.shape) == (2, 1, 64)
