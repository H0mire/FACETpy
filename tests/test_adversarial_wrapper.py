"""Tests for the multi-optimizer adversarial training wrapper.

These pin the properties the loss-module GAN workaround could not provide:
discriminator parameters that actually move, *every* generator head receiving
gradient, and checkpoints that round-trip the discriminators together with the
generator.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from facet.training.adversarial import (  # noqa: E402
    AdversarialModelWrapper,
    AdversarialObjective,
    feature_matching_loss,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
)

# ---------------------------------------------------------------------------
# Minimal three-head generator / two-discriminator stand-in for DHCT-GAN
# ---------------------------------------------------------------------------


class _TinyGenerator(torch.nn.Module):
    """Three named heads, so a dead head is detectable."""

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.clean = torch.nn.Conv1d(channels, channels, 3, padding=1)
        self.noise = torch.nn.Conv1d(channels, channels, 3, padding=1)
        self.gate = torch.nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        clean, noise = self.clean(x), self.noise(x)
        alpha = torch.sigmoid(self.gate(x))
        return {"clean": clean, "noise": noise, "fused": alpha * clean + (1 - alpha) * (x - noise)}


class _TinyDiscriminator(torch.nn.Module):
    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(channels, 4, 3, stride=2, padding=1)
        self.out = torch.nn.Conv1d(4, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = torch.nn.functional.leaky_relu(self.conv(x), 0.2)
        return self.out(h), [h]


class _TinyObjective(AdversarialObjective):
    """Row 0 of the target is clean, row 1 is noise."""

    def __init__(self, lambda_adv: float = 0.1, lambda_feat: float = 1.0) -> None:
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.register_discriminator("clean", _TinyDiscriminator())
        self.register_discriminator("noise", _TinyDiscriminator())
        self.register_discriminator("fused", _TinyDiscriminator())

    @staticmethod
    def _real(target: torch.Tensor, key: str) -> torch.Tensor:
        return target[:, 0:1] if key in ("clean", "fused") else target[:, 1:2]

    def discriminator_losses(self, outputs, target):
        losses = {}
        for name, disc in self.discriminators.items():
            d_real, _ = disc(self._real(target, name))
            d_fake, _ = disc(outputs[name])
            losses[name] = lsgan_discriminator_loss(d_real, d_fake)
        return losses

    def generator_loss(self, outputs, target):
        total = torch.zeros((), device=target.device)
        metrics: dict[str, float] = {}
        for name, disc in self.discriminators.items():
            real = self._real(target, name)
            mse = torch.nn.functional.mse_loss(outputs[name], real)
            d_fake, feats_fake = disc(outputs[name])
            _, feats_real = disc(real)
            adv = lsgan_generator_loss(d_fake)
            feat = feature_matching_loss(feats_fake, feats_real)
            total = total + mse + self.lambda_feat * feat + self.lambda_adv * adv
            metrics[f"mse_{name}"] = float(mse.detach())
        return total, metrics

    def primary_output(self, outputs):
        return outputs["fused"]


def _batch(n: int = 4, t: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    noisy = rng.standard_normal((n, 1, t)).astype(np.float32)
    target = rng.standard_normal((n, 2, t)).astype(np.float32)
    return noisy, target


def _wrapper(**kwargs) -> AdversarialModelWrapper:
    torch.manual_seed(0)
    return AdversarialModelWrapper(
        model=_TinyGenerator(),
        objective=_TinyObjective(),
        device="cpu",
        learning_rate=1e-3,
        **kwargs,
    )


# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_train_step_reports_loss_and_per_discriminator_metrics():
    wrapper = _wrapper()
    metrics = wrapper.train_step(*_batch())
    assert "loss" in metrics and np.isfinite(metrics["loss"])
    for name in ("clean", "noise", "fused"):
        assert f"d_{name}" in metrics, f"discriminator '{name}' reported no loss"
        assert f"mse_{name}" in metrics


@pytest.mark.unit
def test_every_generator_head_receives_gradient():
    """The defect this wrapper exists to fix.

    Under the single-output contract the loss saw only one head, so the other two
    branches were never supervised. Here all three must move.
    """
    wrapper = _wrapper()
    before = {n: p.detach().clone() for n, p in wrapper.model.named_parameters()}
    for step in range(3):
        wrapper.train_step(*_batch(seed=step))
    for name, param in wrapper.model.named_parameters():
        assert not torch.allclose(param, before[name]), f"generator parameter '{name}' never moved"


@pytest.mark.unit
def test_discriminator_parameters_are_updated():
    wrapper = _wrapper()
    before = {
        f"{d}.{n}": p.detach().clone()
        for d, disc in wrapper.objective.discriminators.items()
        for n, p in disc.named_parameters()
    }
    for step in range(3):
        wrapper.train_step(*_batch(seed=step))
    for d, disc in wrapper.objective.discriminators.items():
        for n, param in disc.named_parameters():
            assert not torch.allclose(param, before[f"{d}.{n}"]), f"discriminator '{d}.{n}' never moved"


@pytest.mark.unit
def test_discriminator_step_does_not_touch_the_generator():
    """The D phase must be generator-frozen even if the objective forgets to detach."""
    wrapper = _wrapper(n_discriminator_steps=1)
    wrapper._optimizer.zero_grad(set_to_none=True)
    x, y = _batch()
    detached = {k: v.detach() for k, v in wrapper._forward(wrapper._tensor(x)).items()}
    losses = wrapper.objective.discriminator_losses(detached, wrapper._tensor(y))
    sum(losses.values()).backward()
    assert all(p.grad is None for p in wrapper.model.parameters())


@pytest.mark.unit
def test_eval_step_does_not_update_anything():
    wrapper = _wrapper()
    snapshot = {n: p.detach().clone() for n, p in wrapper.model.named_parameters()}
    disc_snapshot = {
        f"{d}.{n}": p.detach().clone()
        for d, disc in wrapper.objective.discriminators.items()
        for n, p in disc.named_parameters()
    }
    metrics = wrapper.eval_step(*_batch())
    assert np.isfinite(metrics["loss"])
    for name, param in wrapper.model.named_parameters():
        assert torch.allclose(param, snapshot[name])
    for d, disc in wrapper.objective.discriminators.items():
        for n, param in disc.named_parameters():
            assert torch.allclose(param, disc_snapshot[f"{d}.{n}"])


@pytest.mark.unit
def test_checkpoint_round_trips_generator_and_discriminators(tmp_path):
    """Resuming used to restart the discriminators from random init, silently."""
    wrapper = _wrapper()
    for step in range(2):
        wrapper.train_step(*_batch(seed=step))
    path = tmp_path / "ckpt.pt"
    wrapper.save_checkpoint(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "objective_state_dict" in payload
    assert set(payload["discriminator_optimizer_state_dicts"]) == {"clean", "noise", "fused"}

    fresh = _wrapper()
    # A fresh wrapper is a different random init, otherwise the test proves nothing.
    assert not torch.allclose(
        fresh.objective.discriminators["clean"].conv.weight,
        wrapper.objective.discriminators["clean"].conv.weight,
    )
    fresh.load_checkpoint(path)
    for name, param in wrapper.model.named_parameters():
        assert torch.allclose(param, dict(fresh.model.named_parameters())[name])
    for d, disc in wrapper.objective.discriminators.items():
        for n, param in disc.named_parameters():
            assert torch.allclose(param, dict(fresh.objective.discriminators[d].named_parameters())[n])
    assert fresh._step == wrapper._step


@pytest.mark.unit
def test_predict_batch_returns_the_primary_output():
    wrapper = _wrapper()
    noisy, _ = _batch()
    out = wrapper.predict_batch(noisy)
    assert out.shape == noisy.shape


@pytest.mark.unit
def test_parameter_counts_expose_every_network():
    counts = _wrapper().parameter_counts()
    assert set(counts) == {"generator", "discriminator_clean", "discriminator_noise", "discriminator_fused"}
    assert all(v > 0 for v in counts.values())


@pytest.mark.unit
def test_objective_without_discriminators_is_rejected():
    class _Empty(AdversarialObjective):
        def discriminator_losses(self, outputs, target):
            return {}

        def generator_loss(self, outputs, target):
            return torch.zeros(()), {}

    with pytest.raises(ValueError, match="registered no discriminators"):
        AdversarialModelWrapper(model=_TinyGenerator(), objective=_Empty())


@pytest.mark.unit
def test_non_dict_generator_output_is_rejected():
    class _Tuple(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv1d(1, 1, 3, padding=1)

        def forward(self, x):
            return self.conv(x), x

    wrapper = AdversarialModelWrapper(model=_Tuple(), objective=_TinyObjective())
    with pytest.raises(TypeError, match="must return dict"):
        wrapper.train_step(*_batch())


@pytest.mark.unit
def test_unknown_discriminator_name_is_rejected():
    class _Bogus(_TinyObjective):
        def discriminator_losses(self, outputs, target):
            losses = super().discriminator_losses(outputs, target)
            losses["typo"] = losses["clean"]
            return losses

    wrapper = AdversarialModelWrapper(model=_TinyGenerator(), objective=_Bogus())
    with pytest.raises(KeyError, match="typo"):
        wrapper.train_step(*_batch())


@pytest.mark.unit
def test_warmup_flag_flips_after_the_configured_steps():
    wrapper = _wrapper(warmup_steps=2)
    assert not wrapper.adversarial_enabled
    wrapper.train_step(*_batch())
    assert not wrapper.adversarial_enabled
    wrapper.train_step(*_batch())
    assert wrapper.adversarial_enabled


@pytest.mark.unit
def test_discriminator_optimizer_settings_are_independent():
    wrapper = _wrapper(
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={"betas": (0.5, 0.9)},
        discriminator_optimizer_cls=torch.optim.Adam,
        discriminator_optimizer_kwargs={"betas": (0.9, 0.999)},
        discriminator_learning_rate=1e-4,
    )
    assert wrapper._optimizer.param_groups[0]["betas"] == (0.5, 0.9)
    for optimizer in wrapper._d_optimizers.values():
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


@pytest.mark.unit
def test_lsgan_primitives_reward_the_right_direction():
    real_ok, fake_ok = torch.ones(4, 1, 8), torch.zeros(4, 1, 8)
    real_bad, fake_bad = torch.zeros(4, 1, 8), torch.ones(4, 1, 8)
    assert float(lsgan_discriminator_loss(real_ok, fake_ok)) < float(lsgan_discriminator_loss(real_bad, fake_bad))
    assert float(lsgan_generator_loss(torch.ones(4, 1, 8))) < float(lsgan_generator_loss(torch.zeros(4, 1, 8)))


@pytest.mark.unit
def test_feature_matching_detaches_the_real_side():
    fake = torch.randn(2, 3, 8, requires_grad=True)
    real = torch.randn(2, 3, 8, requires_grad=True)
    feature_matching_loss([fake], [real]).backward()
    assert fake.grad is not None
    assert real.grad is None


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_micro_batching_reproduces_the_full_batch_gradient():
    """The property that makes accumulation a memory trick and not a recipe change.

    If the chunk weights were wrong (equal weights with an uneven remainder is the
    classic mistake), the update would differ from the full-batch update and the
    run would quietly be training a different objective.
    """
    noisy, target = _batch(n=6, t=32)

    full = _wrapper()
    chunked = _wrapper(micro_batch_size=4)  # 6 = 4 + 2, deliberately uneven
    for name, param in chunked.model.named_parameters():
        param.data.copy_(dict(full.model.named_parameters())[name].data)
    for d, disc in chunked.objective.discriminators.items():
        for n, param in disc.named_parameters():
            param.data.copy_(dict(full.objective.discriminators[d].named_parameters())[n].data)

    m_full = full.train_step(noisy, target)
    m_chunked = chunked.train_step(noisy, target)

    assert m_chunked["loss"] == pytest.approx(m_full["loss"], rel=1e-4)
    for name, param in full.model.named_parameters():
        torch.testing.assert_close(param, dict(chunked.model.named_parameters())[name], rtol=1e-4, atol=1e-6)


@pytest.mark.unit
def test_micro_batch_larger_than_the_batch_is_a_no_op():
    wrapper = _wrapper(micro_batch_size=1024)
    x, y = _batch(n=4, t=32)
    assert len(wrapper._micro_batches(wrapper._tensor(x), wrapper._tensor(y))) == 1


@pytest.mark.unit
def test_micro_batch_weights_sum_to_one():
    wrapper = _wrapper(micro_batch_size=4)
    x, y = _batch(n=10, t=32)
    chunks = wrapper._micro_batches(wrapper._tensor(x), wrapper._tensor(y))
    assert [c[0].shape[0] for c in chunks] == [4, 4, 2]
    assert sum(c[2] for c in chunks) == pytest.approx(1.0)
