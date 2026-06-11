"""Cheap CPU smoke tests for the paper-accurate ST-GNN model and factories.

All dimensions are an order of magnitude below the production 7/30/512 so
forward + backward on CPU completes in well under a second.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")

from facet.models.st_gnn_paper_accurate_edition.training import (
    NIAZY_PROOF_FIT_CHANNELS,
    SpatiotemporalGNN,
    TemporalGLU,
    build_chebyshev_laplacian,
    build_loss,
    build_model,
)

# Tiny configuration shared across tests.
CONTEXT_EPOCHS = 3
N_CHANNELS = 8
SAMPLES = 64
TINY_CHANNELS = NIAZY_PROOF_FIT_CHANNELS[:N_CHANNELS]
TINY_KWARGS = dict(
    hidden_channels=8,
    bottleneck_channels=4,
    k_order=3,
    time_kernel=3,
    dropout=0.0,
    knn_k=3,
    channel_names=TINY_CHANNELS,
)


def _build_tiny_model(**overrides) -> SpatiotemporalGNN:
    kwargs = dict(TINY_KWARGS)
    kwargs.update(overrides)
    return build_model(input_shape=(CONTEXT_EPOCHS, N_CHANNELS, SAMPLES), **kwargs)


def test_build_chebyshev_laplacian_shape_and_symmetry() -> None:
    l_tilde = build_chebyshev_laplacian(TINY_CHANNELS, k=3)
    assert l_tilde.shape == (N_CHANNELS, N_CHANNELS)
    laplacian = l_tilde + torch.eye(N_CHANNELS)
    assert torch.allclose(laplacian, laplacian.T, atol=1e-6)


def test_build_model_returns_module_with_expected_input_shape() -> None:
    model = _build_tiny_model()
    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, SpatiotemporalGNN)
    assert model.context_epochs == CONTEXT_EPOCHS
    assert model.n_channels == N_CHANNELS
    assert model.samples == SAMPLES


def test_forward_pass_produces_expected_output_shape() -> None:
    model = _build_tiny_model().eval()
    x = torch.randn(2, CONTEXT_EPOCHS, N_CHANNELS, SAMPLES)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, N_CHANNELS, SAMPLES)


def test_causal_variant_forward_shape() -> None:
    model = _build_tiny_model(causal=True).eval()
    x = torch.randn(2, CONTEXT_EPOCHS, N_CHANNELS, SAMPLES)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, N_CHANNELS, SAMPLES)


def test_optimizer_steps_reduce_loss() -> None:
    torch.manual_seed(0)
    model = _build_tiny_model()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = build_loss("mse")
    x = torch.randn(4, CONTEXT_EPOCHS, N_CHANNELS, SAMPLES)
    target = torch.randn(4, N_CHANNELS, SAMPLES)

    model.train()
    losses = []
    for _ in range(5):
        optimiser.zero_grad()
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_glu_gating_is_dauphin_style() -> None:
    # STGCN Eq. 7: GLU = P (linear) * sigmoid(Q), NOT tanh(P) * sigmoid(Q).
    # With a zero-init conv weight and biases, P=0 and sigmoid(0)=0.5 -> 0.
    # We instead verify the formula directly by injecting a known conv.
    torch.manual_seed(0)
    glu = TemporalGLU(in_channels=1, out_channels=2, kernel_size=3)
    x = torch.randn(1, 1, 4, 16)
    with torch.no_grad():
        gated = glu.conv(x)
        p, q = gated.split(2, dim=1)
        expected = p * torch.sigmoid(q)
        actual = glu(x)
    assert torch.allclose(actual, expected, atol=1e-6)
    # And it must NOT match the GTU form tanh(P)*sigmoid(Q).
    gtu = torch.tanh(p) * torch.sigmoid(q)
    assert not torch.allclose(actual, gtu, atol=1e-4)


def test_build_loss_aliases() -> None:
    assert isinstance(build_loss("mse"), torch.nn.MSELoss)
    assert isinstance(build_loss("l1"), torch.nn.L1Loss)
    assert isinstance(build_loss("huber"), torch.nn.SmoothL1Loss)
    with pytest.raises(ValueError):
        build_loss("not-a-loss")


def test_kipf_first_order_variant_builds_and_runs() -> None:
    model = _build_tiny_model(k_order=1).eval()
    x = torch.randn(2, CONTEXT_EPOCHS, N_CHANNELS, SAMPLES)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, N_CHANNELS, SAMPLES)


def test_forward_wrong_shape_raises() -> None:
    model = _build_tiny_model().eval()
    bad = torch.randn(2, CONTEXT_EPOCHS + 2, N_CHANNELS, SAMPLES)
    with pytest.raises((RuntimeError, ValueError)):
        model(bad)


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_torchscript_trace_round_trips() -> None:
    model = _build_tiny_model().eval()
    example = torch.randn(1, CONTEXT_EPOCHS, N_CHANNELS, SAMPLES)
    scripted = torch.jit.trace(model, example)
    with torch.no_grad():
        original = model(example)
        traced = scripted(example)
    assert torch.allclose(original, traced, atol=1e-5)


def test_module_imports_and_registers_unique_processor() -> None:
    module = importlib.import_module("facet.models.st_gnn_paper_accurate_edition")
    assert hasattr(module, "SpatiotemporalGNN")
    assert hasattr(module, "build_model")
    assert hasattr(module, "PaperAccurateSpatiotemporalGNNCorrection")
    correction = module.PaperAccurateSpatiotemporalGNNCorrection
    assert correction.name == "st_gnn_paper_accurate_correction"
