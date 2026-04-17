"""Framework-agnostic TrainableModelWrapper and concrete PyTorch/TF implementations."""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..core import ProcessorValidationError

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TrainableModelWrapper(ABC):
    """Framework-agnostic interface for trainable deep-learning models.

    The :class:`~facet.training.Trainer` only interacts with this interface,
    keeping the training loop completely framework-agnostic.  Concrete
    subclasses translate the numpy arrays to framework tensors, run the
    forward/backward pass, and convert gradients back.

    Parameters
    ----------
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        L2 regularisation coefficient.
    grad_clip_norm : float or None
        Maximum gradient L2 norm.  ``None`` disables clipping.

    Subclassing
    -----------
    1. Override :meth:`train_step` and :meth:`eval_step`.
    2. Override :meth:`save_checkpoint` and :meth:`load_checkpoint`.
    3. Optionally override :meth:`to_inference_adapter` to expose a
       :class:`~facet.correction.DeepLearningModelAdapter` for inference
       after training.

    Example
    -------
    ::

        class MyWrapper(TrainableModelWrapper):
            def train_step(self, noisy, target):
                ...
                return {"loss": float(loss)}

            def eval_step(self, noisy, target):
                ...
                return {"loss": float(val_loss)}

            def save_checkpoint(self, path):
                torch.save(self._model.state_dict(), path)

            def load_checkpoint(self, path):
                self._model.load_state_dict(torch.load(path))
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 1.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm

    @abstractmethod
    def train_step(
        self, noisy: np.ndarray, target: np.ndarray
    ) -> dict[str, float]:
        """Run one forward + backward pass on a mini-batch.

        Parameters
        ----------
        noisy : np.ndarray, shape ``(batch, n_channels, chunk_size)``
        target : np.ndarray, shape ``(batch, n_channels, chunk_size)``

        Returns
        -------
        dict
            Must contain at least the key ``"loss"``.  May include
            additional metrics (e.g. ``"spectral_loss"``).
        """

    @abstractmethod
    def eval_step(
        self, noisy: np.ndarray, target: np.ndarray
    ) -> dict[str, float]:
        """Run one forward pass in evaluation mode (no gradient).

        Same signature and return convention as :meth:`train_step`.
        """

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Serialise model weights + optimiser state to *path*."""

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Deserialise model weights (and optionally optimiser state) from *path*."""

    def to_inference_adapter(self) -> Any:
        """Export the trained model as a FACETpy inference adapter.

        Override in subclasses to enable seamless transition from training
        to inference via the FACETpy correction pipeline.

        Returns
        -------
        DeepLearningModelAdapter
        """
        raise NotImplementedError(
            f"{type(self).__name__}.to_inference_adapter() is not implemented. "
            "Override this method to enable export to a FACETpy inference adapter."
        )

    @property
    def device_info(self) -> str:
        """Human-readable device description (e.g. ``'cuda:0'`` or ``'cpu'``)."""
        return "cpu"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"lr={self.learning_rate}, "
            f"wd={self.weight_decay}, "
            f"grad_clip={self.grad_clip_norm})"
        )


# ---------------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------------


class PyTorchModelWrapper(TrainableModelWrapper):
    """TrainableModelWrapper for PyTorch ``nn.Module`` models.

    Handles the boilerplate of converting numpy arrays to tensors, running
    the forward/backward pass, clipping gradients, and stepping the
    optimiser.

    Parameters
    ----------
    model : torch.nn.Module
        Instantiated model.  Will be moved to *device*.
    loss_fn : callable
        ``(prediction, target) -> scalar_tensor``.  Both arguments are
        ``torch.Tensor`` on *device*.
    device : str
        Target device (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
    optimizer_cls : type, optional
        Optimiser class (default: ``torch.optim.AdamW``).
    optimizer_kwargs : dict, optional
        Extra keyword arguments forwarded to the optimiser constructor
        (besides ``params``, ``lr``, ``weight_decay``).
    scheduler_cls : type, optional
        Learning-rate scheduler class.  Instantiated with the optimiser
        as its first argument plus any *scheduler_kwargs*.
    scheduler_kwargs : dict, optional
        Extra keyword arguments forwarded to the scheduler constructor.
    learning_rate : float
    weight_decay : float
    grad_clip_norm : float or None

    Example
    -------
    ::

        import torch.nn as nn
        import torch.optim as optim

        model = nn.Sequential(
            nn.Conv1d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 4, 3, padding=1),
        )
        loss_fn = nn.MSELoss()

        wrapper = PyTorchModelWrapper(
            model=model,
            loss_fn=loss_fn,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    """

    def __init__(
        self,
        model: Any,
        loss_fn: Callable,
        device: str = "cpu",
        optimizer_cls: type | None = None,
        optimizer_kwargs: dict | None = None,
        scheduler_cls: type | None = None,
        scheduler_kwargs: dict | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = 1.0,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
        )
        if importlib.util.find_spec("torch") is None:
            raise ProcessorValidationError(
                "PyTorchModelWrapper requires torch. Install with: pip install torch"
            )

        import torch

        self._torch = torch
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self._loss_fn = loss_fn

        _opt_cls = optimizer_cls or torch.optim.AdamW
        self._optimizer = _opt_cls(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **(optimizer_kwargs or {}),
        )

        self._scheduler = None
        if scheduler_cls is not None:
            self._scheduler = scheduler_cls(
                self._optimizer, **(scheduler_kwargs or {})
            )

    # ------------------------------------------------------------------
    # TrainableModelWrapper implementation
    # ------------------------------------------------------------------

    def train_step(
        self, noisy: np.ndarray, target: np.ndarray
    ) -> dict[str, float]:
        torch = self._torch
        self._model.train()
        self._optimizer.zero_grad()

        x = torch.as_tensor(noisy, dtype=torch.float32, device=self._device)
        y = torch.as_tensor(target, dtype=torch.float32, device=self._device)

        pred = self._model(x)
        loss = self._loss_fn(pred, y)
        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), self.grad_clip_norm
            )

        self._optimizer.step()

        return {"loss": float(loss.detach().cpu())}

    def eval_step(
        self, noisy: np.ndarray, target: np.ndarray
    ) -> dict[str, float]:
        torch = self._torch
        self._model.eval()

        x = torch.as_tensor(noisy, dtype=torch.float32, device=self._device)
        y = torch.as_tensor(target, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            pred = self._model(x)
            loss = self._loss_fn(pred, y)

        return {"loss": float(loss.cpu())}

    def scheduler_step(self) -> None:
        """Step the LR scheduler (call once per epoch after validation)."""
        if self._scheduler is not None:
            self._scheduler.step()

    def save_checkpoint(self, path: Path) -> None:
        torch = self._torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": (
                    self._scheduler.state_dict()
                    if self._scheduler is not None
                    else None
                ),
            },
            str(path),
        )

    def load_checkpoint(self, path: Path) -> None:
        torch = self._torch
        ckpt = torch.load(str(path), map_location=self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if (
            self._scheduler is not None
            and ckpt.get("scheduler_state_dict") is not None
        ):
            self._scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    @property
    def device_info(self) -> str:
        return str(self._device)

    @property
    def model(self) -> Any:
        """The underlying ``nn.Module``."""
        return self._model


# ---------------------------------------------------------------------------
# TensorFlow / Keras wrapper
# ---------------------------------------------------------------------------


class TensorFlowModelWrapper(TrainableModelWrapper):
    """TrainableModelWrapper for TensorFlow / Keras models.

    Uses the Keras ``train_on_batch`` / ``test_on_batch`` API so that the
    wrapper is compatible with any compiled ``keras.Model``.

    Parameters
    ----------
    model : keras.Model
        A compiled Keras model (must have been compiled with an optimiser
        and loss function via ``model.compile(...)``).
    device : str, optional
        TF device string (e.g. ``"/GPU:0"``).  ``None`` lets TF choose.
    learning_rate : float
        Forwarded to ``TrainableModelWrapper``; the compiled model's
        optimiser learning rate takes precedence at inference time.
    weight_decay : float
    grad_clip_norm : float or None
        Note: gradient clipping must be configured on the Keras optimiser
        directly.  This parameter is stored but not enforced here.

    Example
    -------
    ::

        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu", input_shape=(None, 4)),
            tf.keras.layers.Conv1D(4, 3, padding="same"),
        ])
        model.compile(optimizer="adam", loss="mse")

        wrapper = TensorFlowModelWrapper(model=model)
    """

    def __init__(
        self,
        model: Any,
        device: str | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
        )
        if importlib.util.find_spec("tensorflow") is None:
            raise ProcessorValidationError(
                "TensorFlowModelWrapper requires tensorflow. "
                "Install with: pip install tensorflow"
            )

        import tensorflow as tf

        self._tf = tf
        self._model = model
        self._device = device

    # ------------------------------------------------------------------
    # TrainableModelWrapper implementation
    # ------------------------------------------------------------------

    def _to_tf(self, arr: np.ndarray) -> Any:
        """Convert numpy (batch, ch, time) to TF tensor (batch, time, ch)."""
        # Keras Conv1D expects (batch, timesteps, channels)
        return self._tf.constant(
            np.moveaxis(arr, 1, -1).astype(np.float32)
        )

    def _from_tf(self, tensor: Any) -> np.ndarray:
        """Convert TF tensor (batch, time, ch) back to numpy (batch, ch, time)."""
        arr = tensor.numpy()
        return np.moveaxis(arr, -1, 1)

    def train_step(
        self, noisy: np.ndarray, target: np.ndarray
    ) -> dict[str, float]:
        x = self._to_tf(noisy)
        y = self._to_tf(target)

        if self._device:
            with self._tf.device(self._device):
                result = self._model.train_on_batch(x, y, return_dict=True)
        else:
            result = self._model.train_on_batch(x, y, return_dict=True)

        return {k: float(v) for k, v in result.items()}

    def eval_step(
        self, noisy: np.ndarray, target: np.ndarray
    ) -> dict[str, float]:
        x = self._to_tf(noisy)
        y = self._to_tf(target)

        if self._device:
            with self._tf.device(self._device):
                result = self._model.test_on_batch(x, y, return_dict=True)
        else:
            result = self._model.test_on_batch(x, y, return_dict=True)

        return {k: float(v) for k, v in result.items()}

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # SavedModel format when path has no extension; H5 otherwise
        if path.suffix in {".h5", ".hdf5", ".keras"}:
            self._model.save(str(path))
        else:
            self._model.save(str(path))

    def load_checkpoint(self, path: Path) -> None:
        self._model = self._tf.keras.models.load_model(str(path))

    @property
    def device_info(self) -> str:
        return self._device or "tf-default"

    @property
    def model(self) -> Any:
        """The underlying Keras model."""
        return self._model
