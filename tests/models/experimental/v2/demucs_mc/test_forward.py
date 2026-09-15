"""Shape, gradient flow and equivariance for the multichannel Demucs (run_6 Phase B)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from facet.models.experimental.v2.demucs_mc.training import build_model  # noqa: E402


def _model(ep=7, ch=3, **kw):
    return build_model(input_shape=(ep, ch, 512), depth=3, initial_channels=16, **kw)


@pytest.mark.unit
def test_epochs_are_concatenated_in_time_not_treated_as_units():
    """The whole point of the rebuild: one long waveform per channel.

    A 7-epoch context of 512 samples must reach the encoder as 3584 samples, so
    the strided U-Net keeps a usable bottleneck (14 steps, not 2).
    """
    m = build_model(input_shape=(7, 3, 512), depth=4, initial_channels=8, stride=4)
    seen = {}

    def _hook(_module, inputs, _output):
        seen["t"] = inputs[0].shape[-1]  # returning a value would replace the output

    m.encoder[0].conv.register_forward_hook(_hook)
    m(torch.randn(1, 7, 3, 512))
    assert seen["t"] == 7 * 512


@pytest.mark.unit
def test_forward_shape_matches_the_weg_a_contract():
    m = _model()
    out = m(torch.randn(2, 7, 3, 512))
    assert out.shape == (2, 1, 512)  # target channel only


@pytest.mark.unit
@pytest.mark.parametrize("samples", [256, 500, 512, 576])
def test_forward_handles_lengths_that_do_not_divide_the_stride(samples):
    m = build_model(input_shape=(7, 3, samples), depth=3, initial_channels=16)
    assert m(torch.randn(1, 7, 3, samples)).shape == (1, 1, samples)


@pytest.mark.unit
def test_gradients_reach_every_parameter():
    m = _model()
    m(torch.randn(2, 7, 3, 256)).pow(2).mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing[:5]}"


@pytest.mark.unit
def test_cross_unit_attention_is_permutation_equivariant():
    """Attention adds no positional encoding, so it must not depend on channel order.

    Checked on the attention layer itself: the model's head deliberately *is*
    order-dependent (channel 0 is the target), so end-to-end invariance is not the
    property we want.
    """
    from facet.models.experimental.v2.demucs_mc.training import CrossUnitAttention

    attn = CrossUnitAttention(features=16, n_heads=4).eval()
    x = torch.randn(2, 3, 16, 20)  # (B, channels, features, T)
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        a = attn(x)[:, perm]
        b = attn(x[:, perm])
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


@pytest.mark.unit
def test_model_output_depends_on_the_neighbour_channels():
    """If neighbours were ignored the spatial context would be decorative."""
    m = _model().eval()
    x = torch.randn(1, 7, 3, 256)
    y = x.clone()
    y[:, :, 1:] = torch.randn_like(y[:, :, 1:])  # perturb neighbours only
    with torch.no_grad():
        assert not torch.allclose(m(x), m(y), atol=1e-7)
