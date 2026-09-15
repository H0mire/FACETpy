"""Tests for the strict DHCT-GAN edition.

These pin the claims the README and the paper-accuracy review make, so that the
documentation cannot drift away from the code: three supervised outputs, three
trained discriminators, a genuinely parallel CNN/LGTB encoder, the paper's gating
equation, and a single-channel mode that really is the paper's forward pass rather
than the extension with the extra parts switched off.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facet.models.masterthesis.dhct_gan.strict.training import (  # noqa: E402
    PAPER_DISCRIMINATOR_DIMS,
    CNNLGTBBlock,
    DHCTGanStrictDiscriminator,
    DHCTGanStrictGenerator,
    DHCTGanStrictObjective,
    _CrossChannelBridge,
    build_loss,
    build_model,
    build_wrapper,
)

SMALL_DIMS = (16, 32, 64)


def _generator(epochs: int = 3, channels: int = 3, samples: int = 128, **kwargs) -> DHCTGanStrictGenerator:
    torch.manual_seed(0)
    return build_model(
        context_epochs=epochs,
        n_channels=channels,
        core_samples=samples,
        encoder_dims=SMALL_DIMS,
        **kwargs,
    )


def _batch(n=2, epochs=3, channels=3, samples=128, seed=0):
    rng = np.random.default_rng(seed)
    noisy = rng.standard_normal((n, epochs, channels, samples)).astype(np.float32)
    target = rng.standard_normal((n, 2, samples)).astype(np.float32)  # [artifact, clean]
    return noisy, target


# ---------------------------------------------------------------------------
# Generator structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generator_returns_the_three_paper_outputs():
    out = _generator()(torch.randn(2, 3, 3, 128))
    assert set(out) == {"clean", "noise", "fused"}
    for name, tensor in out.items():
        assert tensor.shape == (2, 1, 128), f"{name} has shape {tuple(tensor.shape)}"


@pytest.mark.unit
def test_fusion_follows_the_paper_gating_equation():
    """Eq. 5: ``Y_pre = mask1 * Y1 + mask2 * (X_raw - Y2)``.

    Reconstructed here from the module's own parts, so a future refactor that
    changes the fusion rule fails instead of quietly redefining the model.
    """
    model = _generator(epochs=1, channels=1, samples=128).eval()
    x = torch.randn(2, 1, 1, 128)
    flat, batch = model._to_paper_shape(x)
    with torch.no_grad():
        # Mirror the forward pass: the branches operate in input-normalised space
        # and the outputs are scaled back to volts.
        scale = model._input_scale(flat)
        normed = flat / scale
        y1 = model.clean_branch(normed)
        y2 = model.noise_branch(normed)
        mask1, mask2 = model.gate(normed)
        expected = (mask1 * y1 + mask2 * (normed - y2)) * scale
        actual = model(x)["fused"]
    torch.testing.assert_close(actual, model._centre_and_mix(expected, batch))


@pytest.mark.unit
def test_gating_masks_are_bounded_by_tanh():
    model = _generator(epochs=1, channels=1, samples=64)
    mask1, mask2 = model.gate(torch.randn(4, 1, 64) * 100)
    assert mask1.abs().max() <= 1.0 and mask2.abs().max() <= 1.0


@pytest.mark.unit
def test_cnn_and_lgtb_run_in_parallel_not_in_sequence():
    """The v2 edition's documented deviation, pinned as a structural property.

    Both paths must see the *block input*. If the LGTB were fed the CNN's output,
    zeroing the CNN would also silence the transformer path; here it must not.
    """
    torch.manual_seed(0)
    block = CNNLGTBBlock(8, 8, n_heads=2, lsa_blocks=4, downsample=False).eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        lgtb_only = block.lgtb(x)
        for parameter in block.cnn.parameters():
            parameter.zero_()
        cnn_out = block.cnn(x)
        still_lgtb = block.lgtb(x)
    assert torch.allclose(cnn_out, torch.zeros_like(cnn_out)), "CNN path was not silenced"
    torch.testing.assert_close(lgtb_only, still_lgtb)  # transformer path is unaffected


@pytest.mark.unit
def test_every_input_epoch_and_channel_reaches_the_output():
    """The absolute constraint: no single-epoch, single-channel forward pass."""
    model = _generator().eval()
    x = torch.randn(2, 3, 3, 128, requires_grad=True)
    model(x)["fused"].abs().sum().backward()
    grad = x.grad.abs().sum(dim=(0, 3))          # (epochs, channels)
    assert (grad.sum(dim=1) > 0).all(), f"an epoch got no gradient: {grad.sum(dim=1)}"
    assert (grad.sum(dim=0) > 0).all(), f"a channel got no gradient: {grad.sum(dim=0)}"


@pytest.mark.unit
def test_single_channel_mode_has_no_extension_parameters():
    """Paper mode must be the paper, not the extension with parts disabled."""
    model = _generator(epochs=1, channels=1, samples=128)
    assert model.describe()["paper_single_channel_mode"] is True
    assert model.cross_channel is None
    assert isinstance(model.channel_mixer, torch.nn.Identity)
    names = [n for n, _ in model.named_parameters()]
    assert not [n for n in names if "cross_channel" in n or "channel_mixer" in n]


@pytest.mark.unit
def test_cross_channel_bridge_is_permutation_equivariant():
    """The bridge treats electrodes as a set, with no learned slot order.

    Tested on the bridge itself, not on the whole generator: the channel mixer
    downstream is deliberately *not* permutation invariant, because the Weg-A
    builder orders the neighbours by geodesic distance and the nearest neighbour
    is legitimately worth more than the second. Asserting invariance end-to-end
    would only be testing the mixer's initial ``[1, 0, 0]`` weights.
    """
    torch.manual_seed(0)
    bridge = _CrossChannelBridge(8, n_heads=2).eval()
    x = torch.randn(2, 3, 8, 16)          # (batch, channels, dim, time)
    perm = [2, 0, 1]
    with torch.no_grad():
        out = bridge(x)
        out_permuted = bridge(x[:, perm])
    torch.testing.assert_close(out[:, perm], out_permuted, rtol=1e-4, atol=1e-6)


@pytest.mark.unit
def test_shape_mismatch_is_rejected_rather_than_reshaped():
    model = _generator()
    with pytest.raises(ValueError, match="epochs x"):
        model(torch.randn(2, 5, 3, 128))
    with pytest.raises(ValueError, match="samples per epoch"):
        model(torch.randn(2, 3, 3, 64))
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(2, 3, 128))


@pytest.mark.unit
@pytest.mark.parametrize("head", ["fc", "conv"])
def test_both_decoder_heads_produce_the_signal_length(head):
    model = _generator(decoder_head=head)
    assert model.describe()["decoder_head"] == head
    assert model(torch.randn(2, 3, 3, 128))["fused"].shape == (2, 1, 128)


@pytest.mark.unit
def test_auto_decoder_head_prefers_the_paper_fc_when_it_is_affordable():
    cheap = _generator(epochs=1, channels=1, samples=128, decoder_head="auto")
    expensive = _generator(epochs=1, channels=1, samples=128, decoder_head="auto", max_fc_head_params=1)
    assert cheap.describe()["decoder_head"] == "fc"
    assert expensive.describe()["decoder_head"] == "conv"


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_discriminator_matches_the_paper_layer_specification():
    disc = DHCTGanStrictDiscriminator()
    convs = [m for m in disc.layers.modules() if isinstance(m, torch.nn.Conv1d)]
    assert len(convs) == 8, "paper Fig. 2a specifies M = 8"
    assert [c.out_channels for c in convs] == list(PAPER_DISCRIMINATOR_DIMS)
    for conv in convs:
        assert conv.kernel_size == (3,) and conv.stride == (2,) and conv.padding == (1,)
    assert len([m for m in disc.layers.modules() if isinstance(m, torch.nn.BatchNorm1d)]) == 8


@pytest.mark.unit
def test_discriminator_exposes_intermediate_features():
    _, features = DHCTGanStrictDiscriminator()(torch.randn(2, 1, 1024))
    assert len(features) >= 4
    assert all(f.ndim == 3 for f in features)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_objective_registers_the_three_paper_discriminators():
    assert set(build_loss().discriminators) == {"clean", "noise", "fused"}


@pytest.mark.unit
def test_objective_pairs_each_discriminator_with_the_right_reference():
    """D2 judges against the artifact, D1/D3 against the clean EEG (§2.2.2)."""
    objective = build_loss()
    target = torch.zeros(1, 2, 8)
    target[0, 0] = 1.0          # artifact row
    target[0, 1] = 2.0          # clean row
    assert objective._real_for("noise", target).mean() == pytest.approx(1.0)
    assert objective._real_for("clean", target).mean() == pytest.approx(2.0)
    assert objective._real_for("fused", target).mean() == pytest.approx(2.0)


@pytest.mark.unit
def test_objective_rejects_a_target_without_the_clean_row():
    objective = build_loss()
    outputs = {k: torch.randn(1, 1, 64) for k in ("clean", "noise", "fused")}
    with pytest.raises(ValueError, match=r"target_extras=\('clean',\)"):
        objective.generator_loss(outputs, torch.randn(1, 1, 64))


@pytest.mark.unit
def test_generator_loss_reports_all_three_paper_terms():
    objective = build_loss()
    outputs = {k: torch.randn(2, 1, 256) for k in ("clean", "noise", "fused")}
    loss, metrics = objective.generator_loss(outputs, torch.randn(2, 2, 256))
    assert torch.isfinite(loss)
    for head in ("clean", "noise", "fused"):
        assert {f"mse_{head}", f"feat_{head}", f"adv_{head}"} <= set(metrics)


@pytest.mark.unit
def test_adversarial_warmup_suppresses_only_the_adversarial_term():
    objective = DHCTGanStrictObjective(adversarial_warmup_steps=2)
    objective.train()
    outputs = {k: torch.randn(2, 1, 256) for k in ("clean", "noise", "fused")}
    target = torch.randn(2, 2, 256)
    _, first = objective.generator_loss(outputs, target)
    assert first["adversarial_on"] == 0.0
    assert all(first[f"adv_{h}"] == 0.0 for h in ("clean", "noise", "fused"))
    objective.generator_loss(outputs, target)
    _, third = objective.generator_loss(outputs, target)
    assert third["adversarial_on"] == 1.0
    assert any(third[f"adv_{h}"] > 0.0 for h in ("clean", "noise", "fused"))


@pytest.mark.unit
def test_primary_output_is_the_gated_signal():
    outputs = {"clean": torch.zeros(1), "noise": torch.ones(1), "fused": torch.full((1,), 2.0)}
    assert build_loss().primary_output(outputs).item() == 2.0


# ---------------------------------------------------------------------------
# End-to-end training step
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_training_step_moves_all_three_branches_and_all_three_discriminators():
    """The defect the strict edition exists to fix, end to end."""
    wrapper = build_wrapper(
        model=_generator(samples=128),
        loss_fn=DHCTGanStrictObjective(discriminator_dims=(16, 32, 64)),
        device="cpu",
        learning_rate=1e-3,
        discriminator_learning_rate=1e-3,
    )
    before = {n: p.detach().clone() for n, p in wrapper.model.named_parameters()}
    disc_before = {
        f"{d}.{n}": p.detach().clone()
        for d, disc in wrapper.objective.discriminators.items()
        for n, p in disc.named_parameters()
    }
    for step in range(2):
        metrics = wrapper.train_step(*_batch(samples=128, seed=step))
    assert np.isfinite(metrics["loss"])

    for prefix in ("clean_branch", "noise_branch", "gate"):
        moved = [
            n
            for n, p in wrapper.model.named_parameters()
            if n.startswith(prefix) and not torch.allclose(p, before[n])
        ]
        assert moved, f"'{prefix}' received no gradient — the dead-head defect is back"
    for d, disc in wrapper.objective.discriminators.items():
        moved = [n for n, p in disc.named_parameters() if not torch.allclose(p, disc_before[f"{d}.{n}"])]
        assert moved, f"discriminator '{d}' was never updated"


@pytest.mark.unit
def test_wrapper_uses_the_paper_adam_betas():
    wrapper = build_wrapper(
        model=_generator(samples=64),
        loss_fn=DHCTGanStrictObjective(discriminator_dims=(8, 16)),
        device="cpu",
    )
    assert wrapper._optimizer.param_groups[0]["betas"] == (0.5, 0.9)
    for optimizer in wrapper._d_optimizers.values():
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)


@pytest.mark.unit
def test_wrapper_rejects_a_plain_loss_module():
    with pytest.raises(TypeError, match="DHCTGanStrictObjective"):
        build_wrapper(model=_generator(samples=64), loss_fn=torch.nn.MSELoss())


# ---------------------------------------------------------------------------
# Amplitude-scale normalisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_loss_terms_are_comparable_at_volt_scale():
    """The defect that wasted the first strict run, as a regression test.

    In volts the reconstruction MSE is ~1e-10 while feature matching and the
    adversarial term are O(1). Without normalisation the paper's lambda1/lambda2
    leave the reconstruction term at ~1e-8 % of the loss and the model optimises
    the discriminators' feature space instead of the signal.
    """
    rng = np.random.default_rng(0)
    volts = lambda s: torch.as_tensor(rng.standard_normal(s) * 1e-3, dtype=torch.float32)  # noqa: E731
    outputs = {k: volts((4, 1, 256)) for k in ("clean", "noise", "fused")}
    target = volts((4, 2, 256))

    unscaled = DHCTGanStrictObjective(discriminator_dims=(16, 32), normalise_scale=False)
    scaled = DHCTGanStrictObjective(discriminator_dims=(16, 32), normalise_scale=True)
    _, m_unscaled = unscaled.generator_loss(outputs, target)
    _, m_scaled = scaled.generator_loss(outputs, target)

    ratio_unscaled = m_unscaled["mse_fused"] / max(m_unscaled["feat_fused"], 1e-12)
    ratio_scaled = m_scaled["mse_fused"] / max(m_scaled["feat_fused"], 1e-12)
    # Normalisation must lift the reconstruction term by orders of magnitude...
    assert ratio_scaled > 1e3 * ratio_unscaled
    # ...into a range where lambda1 = 1.0 actually balances the two terms.
    assert 1e-2 < ratio_scaled < 1e2, ratio_scaled


@pytest.mark.unit
def test_normalisation_is_scale_invariant():
    """Scaling the whole problem by 1000 must not change the loss."""
    rng = np.random.default_rng(1)
    base_out = {k: torch.as_tensor(rng.standard_normal((3, 1, 128)), dtype=torch.float32) for k in
                ("clean", "noise", "fused")}
    base_tgt = torch.as_tensor(rng.standard_normal((3, 2, 128)), dtype=torch.float32)

    torch.manual_seed(0)
    a = DHCTGanStrictObjective(discriminator_dims=(8, 16), lambda_adv=0.0).eval()
    loss_small, _ = a.generator_loss({k: v * 1e-6 for k, v in base_out.items()}, base_tgt * 1e-6)
    loss_large, _ = a.generator_loss({k: v * 1e-3 for k, v in base_out.items()}, base_tgt * 1e-3)
    assert float(loss_small) == pytest.approx(float(loss_large), rel=1e-3)


@pytest.mark.unit
def test_normalisation_does_not_change_model_output_units():
    """Only the loss is normalised; predictions stay in volts, so export is unaffected."""
    model = _generator(samples=64).eval()
    out = model(torch.randn(2, 3, 3, 64) * 1e-3)
    assert out["fused"].abs().mean() < 1.0     # still volt-scale, not standardised


@pytest.mark.unit
def test_scale_divisor_is_detached():
    objective = DHCTGanStrictObjective(discriminator_dims=(8, 16))
    target = torch.randn(2, 2, 64, requires_grad=True) * 1e-3
    scale = objective._scale(target)
    assert not scale.requires_grad


@pytest.mark.unit
def test_all_heads_share_one_loss_scale():
    """Per-head scaling destabilised the clean branch; the scale must be shared.

    With a 56:1 artifact-to-clean ratio, dividing each head's error by its own
    reference RMS amplifies the clean head's error by 56^2. Measured effect: the
    clean branch swung from mse 25 to 3.8e6 within four epochs while the noise
    branch stayed at 1.05. One shared input-derived scale preserves the relative
    magnitudes between heads.
    """
    objective = DHCTGanStrictObjective(discriminator_dims=(8, 16), lambda_adv=0.0)
    target = torch.zeros(2, 2, 64)
    target[:, 0] = 1.0e-3      # artifact, large
    target[:, 1] = 1.0e-3 / 56  # clean, 56x smaller
    scale = objective._scale(target)
    expected = (target[:, 0:1] + target[:, 1:2]).pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
    torch.testing.assert_close(scale, expected)

    # A head predicting zero must not be penalised 56x harder just for having a
    # small reference: the clean head's normalised error is *smaller*, not larger.
    zeros = {k: torch.zeros(2, 1, 64) for k in ("clean", "noise", "fused")}
    _, m = objective.generator_loss(zeros, target)
    assert m["mse_clean"] < m["mse_noise"]


@pytest.mark.unit
def test_generator_output_is_input_scale_equivariant():
    """Normalising inside the generator must make it homogeneous in the input scale.

    The paper's segments are O(1); ours are volts. Working internally in
    normalised space is what lets the decoder's final convolution emit O(1)
    instead of 1e-3, and it makes the model's response proportional to input
    amplitude rather than dependent on it.
    """
    model = _generator(samples=64).eval()
    x = torch.randn(2, 3, 3, 64)
    with torch.no_grad():
        a = model(x)["fused"]
        b = model(x * 1000.0)["fused"]
    torch.testing.assert_close(b, a * 1000.0, rtol=1e-3, atol=1e-9)


@pytest.mark.unit
def test_input_normalisation_can_be_disabled_for_ablation():
    model = _generator(samples=64, input_normalisation=False)
    assert model.describe()["input_normalisation"] is False
    assert model(torch.randn(2, 3, 3, 64)).__class__ is dict
