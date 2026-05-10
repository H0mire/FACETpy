from __future__ import annotations

import numpy as np
import pytest

from facet.models.dpae.training import build_loss, build_model


def test_build_loss_default_is_mse():
    torch = pytest.importorskip("torch")
    loss = build_loss("mse")
    assert isinstance(loss, torch.nn.MSELoss)


def test_dpae_one_batch_backward_updates_gradients():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    model = build_model(input_shape=(1, 64), base_filters=4, latent_filters=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = build_loss("mse")

    noisy = torch.from_numpy(rng.standard_normal((4, 1, 64)).astype(np.float32))
    target = torch.from_numpy(rng.standard_normal((4, 1, 64)).astype(np.float32))

    optimizer.zero_grad()
    pred = model(noisy)
    loss = loss_fn(pred, target)
    loss.backward()

    # At least one trainable parameter must have a non-zero gradient.
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients computed"
    assert any(float(g.abs().max()) > 0 for g in grads), "all gradients are exactly zero"

    optimizer.step()


def test_dpae_build_model_rejects_non_multiple_of_four():
    with pytest.raises(ValueError):
        build_model(input_shape=(1, 63), base_filters=4, latent_filters=8)
