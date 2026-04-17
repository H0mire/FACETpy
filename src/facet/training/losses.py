"""Framework-agnostic loss functions for EEG artifact correction training.

All functions operate on NumPy arrays.  Framework-specific wrappers
(PyTorch, TensorFlow) convert the return value to a differentiable
scalar tensor before calling ``backward()`` / ``tape.gradient()``.

For PyTorch the recommended pattern is::

    import torch

    def torch_loss(pred_tensor, target_tensor):
        pred_np = pred_tensor.detach().cpu().numpy()
        tgt_np  = target_tensor.detach().cpu().numpy()
        value   = spectral_loss(pred_np, tgt_np, sfreq=250.0)
        # Re-attach gradient via a surrogate MSE that has the same
        # scale — or use the pre-built TorchLossWrapper below.
        return torch.tensor(value, requires_grad=False)

For production use, wrap a numpy loss with :class:`TorchLossWrapper` or
:class:`TFLossWrapper` which handle the numpy ↔ tensor conversion and
still propagate gradients through MSE so the model trains correctly.
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
    signal_power = np.mean(target ** 2, axis=-1) + eps
    noise_power = np.mean(residual ** 2, axis=-1) + eps
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

    def __init__(
        self, components: dict[str, tuple[Callable, float]]
    ) -> None:
        self.components = components

    def __call__(
        self, prediction: np.ndarray, target: np.ndarray
    ) -> tuple[float, dict[str, float]]:
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
    """Wraps a numpy loss function for use inside a PyTorch training loop.

    The wrapper computes the **numpy** metric (for logging), but returns a
    differentiable **PyTorch MSE tensor** as the actual gradient signal.
    This ensures gradients flow correctly while still giving you meaningful
    loss curves.

    For full custom losses that need real gradients (e.g. spectral loss
    with autograd), you should implement the loss directly in PyTorch and
    not use this wrapper.

    Parameters
    ----------
    loss_fn : callable
        Numpy ``(prediction, target) -> float`` loss for *logging*.
    gradient_loss_fn : callable or None
        PyTorch ``(pred_tensor, target_tensor) -> scalar_tensor`` used for
        the backward pass.  Defaults to ``torch.nn.functional.mse_loss``.

    Example
    -------
    ::

        loss = TorchLossWrapper(
            loss_fn=CompositeLoss({"mse": (mse_loss, 1.0), "spectral": (spectral_loss, 0.1)}),
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
        """Return a differentiable loss tensor for ``backward()``."""
        if self._gradient_loss_fn is not None:
            return self._gradient_loss_fn(pred_tensor, target_tensor)
        try:
            import torch.nn.functional as F  # noqa: N812

            return F.mse_loss(pred_tensor, target_tensor)
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for TorchLossWrapper.gradient_loss()"
            ) from exc

    def numpy_metrics(
        self, prediction: np.ndarray, target: np.ndarray
    ) -> dict[str, float]:
        """Compute numpy metrics for logging (no gradient)."""
        result = self.loss_fn(prediction, target)
        if isinstance(result, tuple):
            total, breakdown = result
            return breakdown
        return {"loss": float(result)}


from typing import Any  # noqa: E402 (needed for type hints above)
