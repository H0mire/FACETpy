"""Invariants every deployment edition inherits, tested once against a stub core.

The four properties below are the reason the template exists. Each of them fails
silently when wrong — a rescaled output looks like a badly trained model, a
missing demean looks like noise, and an untrained model that deletes the signal
looks like a model that has not converged yet — so each is pinned numerically
rather than left to the editions to get right thirteen times.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from facet.training.deployment_model import DeploymentArtifactModel  # noqa: E402

T, S, B, C = 7, 64, 3, 5
CENTRE = T // 2


class _StubModel(DeploymentArtifactModel):
    """Input ``(B, C, T*S)``; the core is one linear layer over time."""

    def __init__(self, gain: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.core = torch.nn.Conv1d(C, C, kernel_size=1, bias=False)
        torch.nn.init.constant_(self.core.weight, gain)

    def centre_of(self, x):
        return x[..., CENTRE * S:(CENTRE + 1) * S]

    def core_clean(self, x):
        out = self.core(x)
        if self.identity_init:
            out = out + x
        return out[..., CENTRE * S:(CENTRE + 1) * S]


def _input(scale: float = 1.0, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, C, T * S, generator=g) * scale


@pytest.mark.unit
def test_identity_initialisation_changes_nothing():
    """A zero-weight core with the skip must predict *no* artifact.

    Without the skip the same core predicts the whole input as artifact, which is
    the deletion failure — as a starting point, not as something learned.
    """
    x = _input()
    with_skip = _StubModel(gain=0.0, identity_init=True).eval()
    without = _StubModel(gain=0.0, identity_init=False).eval()
    with torch.no_grad():
        assert float(with_skip(x).abs().max()) == pytest.approx(0.0, abs=1e-5)

        centre = x[..., CENTRE * S:(CENTRE + 1) * S]
        centre = centre - centre.mean(-1, keepdim=True)
        predicted = without(x)
        recovered = centre - predicted
        assert float(recovered.pow(2).mean().sqrt()) < 1e-4 * float(centre.pow(2).mean().sqrt())


@pytest.mark.unit
def test_the_output_is_in_the_units_of_the_input():
    """Self-normalisation must be undone: scale the input, scale the output."""
    model = _StubModel(gain=0.3, identity_init=True).eval()
    x = _input()
    with torch.no_grad():
        base = model(x)
        for factor in (1e-6, 1e3):
            assert torch.allclose(model(x * factor), base * factor, rtol=1e-4, atol=1e-9)


@pytest.mark.unit
def test_normalisation_can_be_switched_off_and_then_it_is_not_scale_free():
    model = _StubModel(gain=0.3, identity_init=True, normalise=False).eval()
    x = _input()
    with torch.no_grad():
        assert torch.allclose(model(x * 2.0), model(x) * 2.0, rtol=1e-4, atol=1e-9)


@pytest.mark.unit
def test_every_predicted_epoch_leaves_with_zero_mean():
    """The seam-step failure, made structural instead of remembered."""
    model = _StubModel(gain=0.7, identity_init=True).eval()
    with torch.no_grad():
        out = model(_input() + 42.0)
    assert float(out.mean(dim=-1).abs().max()) < 1e-4 * float(out.abs().max())

    off = _StubModel(gain=0.7, identity_init=True, demean_output=False).eval()
    with torch.no_grad():
        assert float(off(_input() + 42.0).mean(dim=-1).abs().max()) > 0.0


@pytest.mark.unit
def test_a_flat_channel_does_not_blow_up():
    """Dividing by a near-zero standard deviation is a gain of 1/floor."""
    model = _StubModel(gain=0.3, identity_init=True).eval()
    x = _input()
    x[:, 0, :] = 0.0
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()
    assert float(out[:, 0].abs().max()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("packing,shape", [
    ("bcs", (2, 6, S)),
    ("bcts", (2, 6, T * S)),
    ("btcs", (2, T, 6, S)),
])
def test_channel_packings_keep_the_inter_channel_gain(packing, shape):
    """The core must still see which electrode carries more artifact.

    The gradient artifact is one waveform at every electrode times a per-channel
    gain, and that gain is the only thing a multichannel view has that a
    single-channel one does not. Normalising per channel divides it out: every
    electrode arrives at unit variance and the redundancy is gone. It is
    multiplied back onto the output, so the amplitude stays right and nothing in
    the loss ever objects -- which is why this is pinned here rather than left
    to be noticed in a training curve.
    """
    from facet.training.deployment_model import _CONTEXT_DIMS

    x = torch.randn(*shape)
    gains = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    channel_axis = shape.index(6)
    x = x * gains.reshape([-1 if i == channel_axis else 1 for i in range(len(shape))])

    dims = list(_CONTEXT_DIMS[packing])
    normed = (x - x.mean(dim=dims, keepdim=True)) / x.std(dim=dims, keepdim=True)

    rms = normed.pow(2).mean(
        dim=[i for i in range(len(shape)) if i not in (0, channel_axis)]).sqrt()
    spread = float((rms.max(dim=-1).values / rms.min(dim=-1).values).median())
    assert spread > 4.0, (
        f"{packing}: after normalising over {dims} the channels differ by only "
        f"{spread:.2f}x, so the per-channel gain was divided out")


@pytest.mark.unit
def test_normalising_over_the_last_axis_alone_would_erase_the_gain():
    """The counter-example, so the test above cannot pass for the wrong reason."""
    x = torch.randn(2, 6, S) * torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]).reshape(1, -1, 1)
    normed = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)
    rms = normed.pow(2).mean(dim=-1).sqrt()
    assert float((rms.max(dim=-1).values / rms.min(dim=-1).values).median()) < 1.1
