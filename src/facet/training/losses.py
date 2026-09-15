"""Framework-agnostic loss functions for EEG artifact correction training.

All functions in this module operate on NumPy arrays and return plain
floats.  They are convenient to write and test, but NumPy has no
autograd, so they are **not differentiable** — they cannot be used
directly as a gradient signal in a PyTorch/TensorFlow training loop.
Treat them as **metrics** (for logging and monitoring).

To actually train on one of these objectives you need a differentiable
implementation.  Two patterns are supported:

1. **Implement the loss directly in the framework** (recommended).  The
   per-model ``build_loss`` factories under :mod:`facet.models` do this,
   e.g. a differentiable SI-SDR or weighted-MSE ``torch.nn.Module``.  The
   value you log is then exactly the value you optimise.

2. **Wrap a numpy metric with** :class:`TorchLossWrapper`.  The wrapper
   logs the numpy metric and backpropagates a separate, explicitly
   provided differentiable torch loss.  You must supply that gradient
   loss yourself — the wrapper never silently substitutes one (see the
   class docstring).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Numpy primitives
# ---------------------------------------------------------------------------


def mse_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over all elements.

    Parameters
    ----------
    prediction : np.ndarray, shape ``(batch, n_channels, chunk_size)`` or 2-D
    target : np.ndarray, same shape as *prediction*

    Returns
    -------
    float
    """
    return float(np.mean((prediction - target) ** 2))


def mae_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error over all elements."""
    return float(np.mean(np.abs(prediction - target)))


def spectral_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    sfreq: float = 250.0,
    freq_range: tuple[float, float] = (1.0, 45.0),
    nperseg: int = 256,
) -> float:
    """Log-power spectral density (PSD) error in a frequency band.

    Computes the mean absolute difference between the log-PSD of
    *prediction* and *target* in the specified frequency band.  This
    encourages the model to preserve the spectral shape of the clean EEG
    rather than minimising only time-domain amplitude.

    Parameters
    ----------
    prediction : np.ndarray, shape ``(batch, n_channels, chunk_size)``
    target : np.ndarray, same shape
    sfreq : float
        Sampling frequency in Hz.
    freq_range : tuple[float, float]
        Lower and upper frequency boundaries of the band to compare.
    nperseg : int
        Welch segment length.

    Returns
    -------
    float
    """
    from scipy.signal import welch

    pred_flat = prediction.reshape(-1, prediction.shape[-1]).astype(np.float64)
    tgt_flat = target.reshape(-1, target.shape[-1]).astype(np.float64)

    n_seg = min(nperseg, pred_flat.shape[-1])
    freqs, psd_pred = welch(pred_flat, fs=sfreq, nperseg=n_seg, axis=-1)
    _, psd_tgt = welch(tgt_flat, fs=sfreq, nperseg=n_seg, axis=-1)

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not mask.any():
        return 0.0

    log_pred = np.log1p(psd_pred[:, mask])
    log_tgt = np.log1p(psd_tgt[:, mask])
    return float(np.mean(np.abs(log_pred - log_tgt)))


def snr_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Negative signal-to-noise ratio (minimise to maximise SNR).

    SNR = 10 · log10(mean(target²) / mean((prediction - target)²))

    Returns the *negated* mean SNR in dB (so that minimising the loss
    maximises the SNR).

    Parameters
    ----------
    prediction : np.ndarray, shape ``(batch, n_channels, chunk_size)``
    target : np.ndarray, same shape
    eps : float
        Small constant for numerical stability.
    """
    residual = prediction - target
    signal_power = np.mean(target**2, axis=-1) + eps
    noise_power = np.mean(residual**2, axis=-1) + eps
    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return float(-np.mean(snr_db))


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


class CompositeLoss:
    """Weighted sum of multiple named loss functions.

    Each component loss is a callable with signature
    ``(prediction, target) -> float``.

    Parameters
    ----------
    components : dict[str, tuple[Callable, float]]
        Mapping from component name to ``(loss_fn, weight)`` pairs.

    Example
    -------
    ::

        loss = CompositeLoss({
            "mse":      (mse_loss, 1.0),
            "spectral": (spectral_loss, 0.1),
        })
        value, breakdown = loss(pred, target)
        # value   → 0.012  (weighted total)
        # breakdown → {"mse": 0.010, "spectral": 0.020, "total": 0.012}
    """

    def __init__(self, components: dict[str, tuple[Callable, float]]) -> None:
        self.components = components

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute weighted total loss and per-component breakdown.

        Returns
        -------
        total : float
        breakdown : dict[str, float]
            Individual component values (unweighted) plus ``"total"``.
        """
        total = 0.0
        breakdown: dict[str, float] = {}
        for name, (fn, weight) in self.components.items():
            val = fn(prediction, target)
            breakdown[name] = val
            total += weight * val
        breakdown["total"] = total
        return total, breakdown


# ---------------------------------------------------------------------------
# Framework shims (gradient-compatible wrappers)
# ---------------------------------------------------------------------------


class TorchLossWrapper:
    """Decouples a numpy *logging* metric from the *differentiable* gradient.

    The wrapper computes the **numpy** ``loss_fn`` for logging (meaningful
    loss curves, per-component breakdown) and backpropagates a **separate,
    explicitly provided** differentiable torch loss (``gradient_loss_fn``).

    Because the numpy ``loss_fn`` is non-differentiable it can never be the
    gradient signal.  The wrapper therefore does **not** invent one: if no
    ``gradient_loss_fn`` is given, :meth:`gradient_loss` raises ``ValueError``
    instead of silently optimising MSE while you watch a different curve.
    This avoids the trap where the logged loss diverges from the optimised
    loss.

    For most cases the simpler, leak-free approach is to implement the loss
    directly as a differentiable ``torch.nn.Module`` (see the per-model
    ``build_loss`` factories) so the logged and optimised loss are identical.

    Parameters
    ----------
    loss_fn : callable
        Numpy ``(prediction, target) -> float`` (or :class:`CompositeLoss`)
        used for *logging only*.
    gradient_loss_fn : callable or None
        Differentiable torch ``(pred_tensor, target_tensor) -> scalar_tensor``
        used for the backward pass.  Required before :meth:`gradient_loss`
        can be called; if ``None`` it raises ``ValueError``.  Pass
        ``torch.nn.functional.mse_loss`` explicitly if you do want an MSE
        gradient.

    Example
    -------
    ::

        import torch.nn.functional as F

        loss = TorchLossWrapper(
            loss_fn=CompositeLoss({"mse": (mse_loss, 1.0), "spectral": (spectral_loss, 0.1)}),
            gradient_loss_fn=F.mse_loss,  # explicit: this is the signal we optimise
        )

        # Inside train_step:
        grad_loss = loss.gradient_loss(pred_tensor, target_tensor)
        grad_loss.backward()
        metrics = loss.numpy_metrics(pred_np, target_np)  # for logging
    """

    def __init__(
        self,
        loss_fn: Callable,
        gradient_loss_fn: Callable | None = None,
    ) -> None:
        self.loss_fn = loss_fn
        self._gradient_loss_fn = gradient_loss_fn

    def gradient_loss(self, pred_tensor: Any, target_tensor: Any) -> Any:
        """Return the differentiable loss tensor for ``backward()``.

        Raises
        ------
        ValueError
            If no ``gradient_loss_fn`` was provided.  The numpy ``loss_fn``
            is non-differentiable and is used for logging only, so there is
            no safe gradient to fall back on.
        """
        if self._gradient_loss_fn is None:
            raise ValueError(
                "TorchLossWrapper.gradient_loss() requires an explicit `gradient_loss_fn` "
                "(a differentiable torch loss). The numpy `loss_fn` is non-differentiable and "
                "is used for logging only. Pass gradient_loss_fn=torch.nn.functional.mse_loss "
                "for an MSE gradient, or implement your loss directly as a torch.nn.Module."
            )
        return self._gradient_loss_fn(pred_tensor, target_tensor)

    def numpy_metrics(self, prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
        """Compute numpy metrics for logging (no gradient)."""
        result = self.loss_fn(prediction, target)
        if isinstance(result, tuple):
            total, breakdown = result
            return breakdown
        return {"loss": float(result)}


from typing import Any  # noqa: E402 (needed for type hints above)
