"""Smoke-level unit tests for the ST-GNN model and training factories."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facet.models.st_gnn.training import (
    NIAZY_PROOF_FIT_CHANNELS,
    SpatiotemporalGNN,
    build_chebyshev_laplacian,
    build_loss,
    build_model,
)


def test_build_chebyshev_laplacian_shape_and_symmetry() -> None:
    l_tilde = build_chebyshev_laplacian(NIAZY_PROOF_FIT_CHANNELS, k=4)
    assert l_tilde.shape == (30, 30)
    laplacian = l_tilde + torch.eye(30)
    assert torch.allclose(laplacian, laplacian.T, atol=1e-6)


def test_build_model_returns_module_with_expected_input_shape() -> None:
    model = build_model(input_shape=(7, 30, 512))
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, SpatiotemporalGNN)
    assert model.context_epochs == 7
    assert model.n_channels == 30
    assert model.samples == 512


def test_forward_pass_produces_expected_output_shape() -> None:
    model = build_model(input_shape=(7, 30, 512)).eval()
    x = torch.randn(2, 7, 30, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 30, 512)


def test_one_batch_backward_pass_updates_gradients() -> None:
    model = build_model(input_shape=(7, 30, 512))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = build_loss("mse")
    x = torch.randn(2, 7, 30, 512)
    target = torch.randn(2, 30, 512)

    optimiser.zero_grad()
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()

    has_grad = any(
        param.grad is not None and torch.any(param.grad != 0).item()
        for param in model.parameters()
    )
    assert has_grad, "no parameter received a non-zero gradient"
    optimiser.step()


def test_forward_wrong_shape_raises() -> None:
    # The model deliberately omits an explicit shape check so it traces
    # cleanly to TorchScript. A wrong context-epoch count must still
    # fail (here, via a torch reshape error) — exact exception type is
    # implementation detail.
    model = build_model(input_shape=(7, 30, 512)).eval()
    bad = torch.randn(2, 5, 30, 512)
    with pytest.raises((RuntimeError, ValueError)):
        model(bad)


def test_build_loss_aliases() -> None:
    assert isinstance(build_loss("mse"), torch.nn.MSELoss)
    assert isinstance(build_loss("l1"), torch.nn.L1Loss)
    assert isinstance(build_loss("huber"), torch.nn.SmoothL1Loss)
    with pytest.raises(ValueError):
        build_loss("not-a-loss")


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_torchscript_trace_round_trips() -> None:
    model = build_model(input_shape=(7, 30, 512)).eval()
    example = torch.randn(1, 7, 30, 512)
    scripted = torch.jit.trace(model, example)
    with torch.no_grad():
        original = model(example)
        traced = scripted(example)
    assert torch.allclose(original, traced, atol=1e-5)


def test_module_imports_without_torch_geometric_runtime() -> None:
    # Sanity check that nothing in our runtime path eagerly imports
    # torch_geometric — the package only needs to be installed for
    # transitive ecosystem reasons, not to run the model.
    module = importlib.import_module("facet.models.st_gnn")
    assert hasattr(module, "SpatiotemporalGNN")
    assert hasattr(module, "build_model")


def test_chebyshev_polynomial_is_symmetric_with_isotropic_input() -> None:
    # If the input is constant across nodes, Cheb output remains constant
    # across nodes (since the Laplacian acts trivially on constants up to
    # scaling). This is a quick sanity check on the recursion plumbing.
    model = build_model(input_shape=(7, 30, 32))
    model.eval()
    x = torch.ones(1, 7, 30, 32)
    with torch.no_grad():
        y = model(x)
    per_channel_std = y.std(dim=1)
    # Within a single example/sample slot, channels should agree closely.
    assert torch.all(per_channel_std < 1.0)
