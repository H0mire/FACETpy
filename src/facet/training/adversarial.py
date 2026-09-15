"""Multi-optimizer adversarial training for ``facet-train``.

Why this module exists
----------------------
The standard :class:`~facet.training.wrapper.PyTorchModelWrapper` implements one
contract: *one* forward returning *one* tensor, *one* loss, *one* optimizer. Every
GAN in this repository has been bent to fit it, and the bend is documented as a
caveat in ``docs/source/development/deep_learning_model_audit.md``:

* The discriminators had to be hidden **inside the loss module**, where they hold
  private optimizers and run their own update when ``loss_fn(pred, target)`` is
  called. Their parameters are therefore invisible to the trainer: they are not in
  ``model.parameters()``, so they are not gradient-clipped, not checkpointed, and
  not restored on resume. A run that stops and resumes silently restarts its
  discriminators from scratch.
* Worse, the loss only ever receives the *single* tensor the generator returned.
  DHCT-GAN's generator has three paper outputs (branch-1 clean, branch-2 noise,
  fused clean) and three discriminators, one per output. With one visible output,
  two of the three branches get no gradient at all — the audit records this as the
  reason the paper's three-branch logic is "nicht vollstaendig realisiert".

This module removes the bend instead of documenting it again. The wrapper owns the
*mechanics* every GAN shares — alternating updates, one optimizer per network,
gradient clipping per network, complete checkpoints, per-component metrics — and
delegates the *objective* to an :class:`AdversarialObjective` supplied by the model
package, because the loss algebra (which discriminator judges which output against
which target, LSGAN vs. hinge, feature matching, gradient balancing) is exactly the
part that differs per paper and must not be guessed centrally.

Contract
--------
The generator returns a ``dict[str, Tensor]`` of named outputs. Naming them, rather
than returning a tuple, is what lets the objective and the checkpoint stay readable
when a paper has three or four heads.

Usage from a training YAML::

    model:
      framework: pytorch
      factory: facet.models.<pkg>.training:build_model
      wrapper_factory: facet.models.<pkg>.training:build_wrapper
      wrapper_kwargs: {...}

See :mod:`facet.models.masterthesis.dhct_gan.strict.training` for a worked example.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .wrapper import TrainableModelWrapper

__all__ = [
    "AdversarialObjective",
    "AdversarialModelWrapper",
    "lsgan_discriminator_loss",
    "lsgan_generator_loss",
    "feature_matching_loss",
]


# ---------------------------------------------------------------------------
# Reusable GAN loss primitives
# ---------------------------------------------------------------------------


def lsgan_discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN discriminator objective ``0.5*E[D(G(x))^2] + 0.5*E[(D(y)-1)^2]``.

    This is DHCT-GAN Eq. 13. Least squares rather than BCE because the saturating
    log-loss gives almost no gradient once the discriminator wins, which on a
    strongly-structured signal like a gradient artifact happens within a few
    hundred steps.
    """
    return 0.5 * d_fake.pow(2).mean() + 0.5 * (d_real - 1.0).pow(2).mean()


def lsgan_generator_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN generator objective ``E[(D(G(x)) - 1)^2]`` (DHCT-GAN Eq. 12)."""
    return (d_fake - 1.0).pow(2).mean()


def feature_matching_loss(
    features_fake: list[torch.Tensor],
    features_real: list[torch.Tensor],
) -> torch.Tensor:
    """MSE between discriminator intermediate features (DHCT-GAN Eq. 11).

    The real-side features are detached: feature matching is meant to pull the
    generator towards the real activations, not to drag the reference towards the
    generator.
    """
    if not features_fake:
        return torch.zeros((), device=features_real[0].device if features_real else None)
    total = torch.zeros((), device=features_fake[0].device)
    for fake, real in zip(features_fake, features_real, strict=True):
        total = total + torch.nn.functional.mse_loss(fake, real.detach())
    return total / len(features_fake)


# ---------------------------------------------------------------------------
# Objective interface
# ---------------------------------------------------------------------------


class AdversarialObjective(ABC, torch.nn.Module):
    """The paper-specific half of an adversarial run.

    A subclass owns the discriminators and states, in code, exactly which output is
    judged against which target and with what weights. The wrapper never guesses
    any of that; it only decides *when* each network is updated.

    Subclasses must register their discriminators in :attr:`discriminators` so the
    wrapper can give each one its own optimizer and checkpoint slot. Anything left
    out of that mapping is trained by nobody.
    """

    #: name -> discriminator module. Populated by the subclass' ``__init__``.
    discriminators: dict[str, torch.nn.Module]

    def __init__(self) -> None:
        super().__init__()
        self.discriminators = {}

    def register_discriminator(self, name: str, module: torch.nn.Module) -> torch.nn.Module:
        """Register *module* as a discriminator and as a child module."""
        if name in self.discriminators:
            raise ValueError(f"Discriminator '{name}' is already registered")
        self.add_module(f"disc_{name}", module)
        self.discriminators[name] = module
        return module

    @abstractmethod
    def discriminator_losses(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return one scalar loss per discriminator name.

        *outputs* are already detached by the wrapper, so a subclass cannot leak a
        discriminator update back into the generator by accident.
        """

    @abstractmethod
    def generator_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(scalar_loss, extra_metrics)`` for the generator update."""

    def primary_output(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """The tensor that counts as *the* prediction for export and callbacks.

        Defaults to the first entry; override when the exported quantity is not the
        first head (for DHCT-GAN it is the fused output, not branch 1).
        """
        return next(iter(outputs.values()))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class AdversarialModelWrapper(TrainableModelWrapper):
    """Trainer-visible wrapper for a generator plus N discriminators.

    One step is the alternation every GAN paper in this repo describes: with the
    generator frozen, each discriminator is updated on its own loss against
    detached generator outputs; then, with the discriminators frozen, the generator
    is updated on the combined objective.

    Every network gets its own optimizer, its own gradient clipping and its own
    checkpoint entry, which is the part the loss-module workaround could not do.

    Parameters
    ----------
    model
        Generator. Its ``forward`` must return ``dict[str, Tensor]``.
    objective
        :class:`AdversarialObjective` holding the discriminators and the loss
        algebra.
    device
        Torch device string.
    optimizer_cls, optimizer_kwargs
        Generator optimizer (default ``AdamW``). DHCT-GAN uses Adam with
        ``betas=(0.5, 0.9)`` for G, so this must be configurable per network.
    discriminator_optimizer_cls, discriminator_optimizer_kwargs
        Discriminator optimizer; falls back to the generator's when unset.
    discriminator_learning_rate
        Discriminator LR; falls back to *learning_rate* when unset.
    discriminator_weight_decay
        Discriminator weight decay; falls back to *weight_decay* when unset.
    n_discriminator_steps
        Discriminator updates per generator update (paper default 1).
    micro_batch_size
        Split each batch into pieces of at most this size, accumulate gradients
        across them and step once. This decouples the batch size the GPU can hold
        from the batch size the optimizer sees, which matters when a paper
        specifies one (DHCT-GAN: 40) and the model does not fit it. ``None``
        disables splitting. Losses are weighted by ``chunk / batch`` so the update
        equals the full-batch update rather than approximating it.
    warmup_steps
        Number of initial steps during which the discriminators are updated but the
        generator's adversarial term is suppressed by the objective. Exposed as
        :attr:`adversarial_enabled` for the objective to read; a cold discriminator
        emits noise, and on a 40 dB-dynamic-range artifact that noise is enough to
        derail the reconstruction term before it has converged.
    scheduler_cls, scheduler_kwargs
        Optional LR scheduler for the generator optimizer only.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        objective: AdversarialObjective,
        device: str = "cpu",
        optimizer_cls: type | None = None,
        optimizer_kwargs: dict | None = None,
        discriminator_optimizer_cls: type | None = None,
        discriminator_optimizer_kwargs: dict | None = None,
        discriminator_learning_rate: float | None = None,
        discriminator_weight_decay: float | None = None,
        n_discriminator_steps: int = 1,
        micro_batch_size: int | None = None,
        warmup_steps: int = 0,
        scheduler_cls: type | None = None,
        scheduler_kwargs: dict | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = 1.0,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
        )
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self._objective = objective.to(self._device)
        self._n_d_steps = max(0, int(n_discriminator_steps))
        self._micro_batch_size = int(micro_batch_size) if micro_batch_size else None
        self._warmup_steps = max(0, int(warmup_steps))
        self._step = 0

        if not self._objective.discriminators:
            raise ValueError(
                "AdversarialObjective registered no discriminators. Use PyTorchModelWrapper "
                "for a purely deterministic loss, or register the discriminators the paper "
                "specifies via AdversarialObjective.register_discriminator()."
            )

        g_cls = optimizer_cls or torch.optim.AdamW
        self._optimizer = g_cls(
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **(optimizer_kwargs or {}),
        )

        d_cls = discriminator_optimizer_cls or g_cls
        d_lr = learning_rate if discriminator_learning_rate is None else float(discriminator_learning_rate)
        d_wd = weight_decay if discriminator_weight_decay is None else float(discriminator_weight_decay)
        d_kwargs = discriminator_optimizer_kwargs if discriminator_optimizer_kwargs is not None else {}
        self._d_optimizers: dict[str, torch.optim.Optimizer] = {
            name: d_cls(disc.parameters(), lr=d_lr, weight_decay=d_wd, **d_kwargs)
            for name, disc in self._objective.discriminators.items()
        }

        self._scheduler = None
        if scheduler_cls is not None:
            self._scheduler = scheduler_cls(self._optimizer, **(scheduler_kwargs or {}))

    # ------------------------------------------------------------------
    # State the objective may read
    # ------------------------------------------------------------------

    @property
    def adversarial_enabled(self) -> bool:
        """``False`` while the discriminators are still warming up."""
        return self._step >= self._warmup_steps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self._device)

    def _forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._model(x)
        if not isinstance(outputs, dict):
            raise TypeError(
                f"{type(self._model).__name__}.forward() must return dict[str, Tensor] under "
                f"AdversarialModelWrapper, got {type(outputs).__name__}. Name the heads so the "
                "objective and the checkpoint stay readable."
            )
        return outputs

    def _clip(self, parameters) -> None:
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm)

    # ------------------------------------------------------------------
    # TrainableModelWrapper implementation
    # ------------------------------------------------------------------

    def _micro_batches(self, x: torch.Tensor, y: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """Split a batch into memory-sized pieces plus each piece's loss weight.

        Weights are ``chunk / batch`` so that summing the scaled mean-reduced
        losses reproduces the full-batch gradient exactly. Without that scaling a
        remainder chunk would be over-weighted, which is the classic way gradient
        accumulation silently changes the objective.
        """
        n = x.shape[0]
        size = self._micro_batch_size or n
        if size >= n:
            return [(x, y, 1.0)]
        return [(x[i : i + size], y[i : i + size], min(size, n - i) / n) for i in range(0, n, size)]

    def train_step(self, noisy: np.ndarray, target: np.ndarray) -> dict[str, float]:
        self._model.train()
        self._objective.train()
        x, y = self._tensor(noisy), self._tensor(target)
        metrics: dict[str, float] = {}
        chunks = self._micro_batches(x, y)

        # ---- Discriminator phase: generator frozen -------------------------
        # The generator outputs are detached here rather than in the objective, so
        # a subclass cannot accidentally push discriminator gradients into the
        # generator. With micro-batching, gradients accumulate across chunks and
        # each discriminator steps once, so the update is the full-batch update.
        if self._n_d_steps > 0:
            for _ in range(self._n_d_steps):
                for optimizer in self._d_optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                totals: dict[str, float] = {}
                for xc, yc, weight in chunks:
                    with torch.no_grad():
                        detached = {k: v.detach() for k, v in self._forward(xc).items()}
                    d_losses = self._objective.discriminator_losses(detached, yc)
                    for name, d_loss in d_losses.items():
                        if name not in self._d_optimizers:
                            raise KeyError(
                                f"discriminator_losses() returned '{name}', which is not a registered "
                                f"discriminator. Registered: {sorted(self._d_optimizers)}"
                            )
                        (d_loss * weight).backward()
                        totals[name] = totals.get(name, 0.0) + float(d_loss.detach().cpu()) * weight
                for name, optimizer in self._d_optimizers.items():
                    self._clip(self._objective.discriminators[name].parameters())
                    optimizer.step()
                metrics |= {f"d_{k}": v for k, v in totals.items()}

        # ---- Generator phase: discriminators frozen ------------------------
        # The discriminators keep requires_grad=True so feature matching stays
        # differentiable w.r.t. the generator; they simply are not stepped, and
        # their .grad is cleared afterwards so the next D phase starts clean.
        self._optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        extras: dict[str, float] = {}
        for xc, yc, weight in chunks:
            outputs = self._forward(xc)
            g_loss, extra = self._objective.generator_loss(outputs, yc)
            (g_loss * weight).backward()
            total_loss += float(g_loss.detach().cpu()) * weight
            for k, v in extra.items():
                extras[k] = extras.get(k, 0.0) + float(v) * weight
        self._clip(self._model.parameters())
        self._optimizer.step()
        for optimizer in self._d_optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        self._step += 1
        metrics["loss"] = total_loss
        metrics.update(extras)
        return metrics

    def eval_step(self, noisy: np.ndarray, target: np.ndarray) -> dict[str, float]:
        self._model.eval()
        self._objective.eval()
        x, y = self._tensor(noisy), self._tensor(target)
        with torch.no_grad():
            outputs = self._forward(x)
            g_loss, extra = self._objective.generator_loss(outputs, y)
            d_losses = self._objective.discriminator_losses({k: v for k, v in outputs.items()}, y)
        metrics = {"loss": float(g_loss.cpu())}
        metrics.update({f"d_{k}": float(v.cpu()) for k, v in d_losses.items()})
        metrics.update({k: float(v) for k, v in extra.items()})
        return metrics

    def predict_batch(self, noisy: np.ndarray) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            outputs = self._forward(self._tensor(noisy))
            primary = self._objective.primary_output(outputs)
        return primary.detach().cpu().numpy()

    def scheduler_step(self) -> None:
        if self._scheduler is not None:
            self._scheduler.step()

    def save_checkpoint(self, path: Path) -> None:
        """Save generator, every discriminator, and every optimizer.

        The loss-module workaround saved only the generator, so resuming a GAN run
        restarted its discriminators from random init against a trained generator —
        a large, silent perturbation exactly at the resume point.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": (self._scheduler.state_dict() if self._scheduler is not None else None),
                "objective_state_dict": self._objective.state_dict(),
                "discriminator_optimizer_state_dicts": {
                    name: optimizer.state_dict() for name, optimizer in self._d_optimizers.items()
                },
                "adversarial_step": self._step,
            },
            str(path),
        )

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)
        self._model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("objective_state_dict"):
            self._objective.load_state_dict(ckpt["objective_state_dict"])
        for name, state in (ckpt.get("discriminator_optimizer_state_dicts") or {}).items():
            if name in self._d_optimizers:
                self._d_optimizers[name].load_state_dict(state)
        if self._scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self._scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._step = int(ckpt.get("adversarial_step", 0))

    @property
    def device_info(self) -> str:
        return str(self._device)

    @property
    def model(self) -> Any:
        """The generator — what ``facet-train`` exports for inference."""
        return self._model

    @property
    def objective(self) -> AdversarialObjective:
        return self._objective

    def parameter_counts(self) -> dict[str, int]:
        """Trainable parameters per network, for the run report."""
        counts = {"generator": sum(p.numel() for p in self._model.parameters() if p.requires_grad)}
        for name, disc in self._objective.discriminators.items():
            counts[f"discriminator_{name}"] = sum(p.numel() for p in disc.parameters() if p.requires_grad)
        return counts

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._objective.discriminators))
        return (
            f"AdversarialModelWrapper(generator={type(self._model).__name__}, "
            f"discriminators=[{names}], lr={self.learning_rate}, device={self._device})"
        )
