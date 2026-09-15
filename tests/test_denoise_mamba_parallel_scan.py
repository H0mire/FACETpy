"""The parallel scan must compute what the sequential loop computes.

The whole justification for replacing a 512-step loop with a log-depth scan is
that it is the *same* recurrence — affine maps compose associatively, so only the
order of the floating-point additions changes. If that stops being true the
edition is no longer DenoiseMamba, and nothing else in the pipeline would notice.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from facet.models.denoise_mamba_deployment_edition.training import (  # noqa: E402
    ParallelScanSSM,
    build_model,
    parallel_affine_scan,
)


def _sequential(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    state = torch.zeros(a.shape[0], *a.shape[2:], dtype=a.dtype, device=a.device)
    out = []
    for t in range(a.shape[1]):
        state = a[:, t] * state + b[:, t]
        out.append(state)
    return torch.stack(out, dim=1)


@pytest.mark.unit
@pytest.mark.parametrize("length", [1, 2, 7, 8, 64, 512])
def test_scan_matches_the_sequential_recurrence(length):
    """Non-powers of two included: the shift's identity fill is what makes those
    exact, and an off-by-one there would only show up at the boundaries."""
    torch.manual_seed(0)
    a = torch.rand(3, length, 5, 4) * 0.99 + 0.005      # the model's range: (0, 1)
    b = torch.randn(3, length, 5, 4)
    expected = _sequential(a, b)
    got = parallel_affine_scan(a, b)
    assert got.shape == expected.shape
    scale = float(expected.abs().max())
    assert float((got - expected).abs().max()) < 1e-5 * max(scale, 1.0)


@pytest.mark.unit
def test_scan_gradients_match_too():
    torch.manual_seed(1)
    a = (torch.rand(2, 32, 3, 4) * 0.9 + 0.05)
    b = torch.randn(2, 32, 3, 4)
    grads = []
    for fn in (_sequential, parallel_affine_scan):
        aa, bb = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        fn(aa, bb).square().sum().backward()
        grads.append((aa.grad.clone(), bb.grad.clone()))
    for g_seq, g_par in zip(grads[0], grads[1], strict=True):
        assert torch.allclose(g_seq, g_par, atol=1e-4, rtol=1e-3)


@pytest.mark.unit
def test_the_whole_model_agrees_with_the_sequential_one():
    """Same seed, same weights, same output — the swap re-parents parameters
    rather than rebuilding, so this compares the identical network."""
    torch.manual_seed(0)
    seq = build_model(parallel_scan=False).eval()
    torch.manual_seed(0)
    par = build_model(parallel_scan=True).eval()

    x = torch.randn(4, 1, 512) * 1e3
    with torch.no_grad():
        a, b = seq(x), par(x)
    rel = float((a - b).abs().max()) / float(a.abs().max())
    assert rel < 1e-5, f"relative difference {rel:.3e}"


@pytest.mark.unit
def test_a_state_dict_moves_between_the_two_implementations():
    """A checkpoint trained with one must load into the other, or the speed-up
    silently forks the model zoo."""
    from facet.models.denoise_mamba.training import SelectiveSSM

    old = SelectiveSSM(d_inner=8, d_state=4)
    new = ParallelScanSSM(d_inner=8, d_state=4)
    new.load_state_dict(old.state_dict(), strict=True)
    old.load_state_dict(new.state_dict(), strict=True)

    x = torch.randn(2, 16, 8)
    with torch.no_grad():
        assert torch.allclose(old(x), new(x), atol=1e-5)


@pytest.mark.unit
def test_checkpointing_changes_neither_output_nor_gradients():
    """Recomputing the scan in the backward pass must be invisible in the result.

    Dropout has to be off for the comparison to mean anything: both models are in
    train mode so the checkpoint path is actually taken, and two train-mode
    forwards otherwise draw different masks — which is what this test caught the
    first time it was written.
    """
    def make(checkpoint: bool):
        torch.manual_seed(0)
        model = build_model(parallel_scan=True, checkpoint_scan=checkpoint).train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        return model

    x = torch.randn(2, 1, 512)
    outputs, grads = [], []
    for checkpoint in (True, False):
        model = make(checkpoint)
        out = model(x)
        out.square().sum().backward()
        outputs.append(out.detach().clone())
        grads.append(torch.cat([p.grad.reshape(-1) for p in model.parameters()
                                if p.grad is not None]))

    assert torch.allclose(outputs[0], outputs[1], atol=1e-5)
    assert torch.allclose(grads[0], grads[1], atol=1e-4, rtol=1e-3)
