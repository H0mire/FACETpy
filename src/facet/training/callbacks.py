"""Training callbacks — checkpoint saving, early stopping, and extensibility hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from .trainer import TrainingState


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Callback:
    """Base class for all training callbacks.

    Subclass and override any combination of hook methods.  The
    :class:`~facet.training.Trainer` calls these hooks at well-defined
    points during the training loop.

    Hook execution order per epoch::

        on_train_begin          (once, before epoch 1)
          └─ on_epoch_begin     (start of each epoch)
               └─ on_batch_begin  (start of each mini-batch)
               └─ on_batch_end    (end of each mini-batch)
          └─ on_epoch_end       (after each epoch, after validation)
        on_train_end            (once, after the last epoch)

    The ``state`` argument is a :class:`~facet.training.trainer.TrainingState`
    dataclass with the following read attributes:

    * ``epoch`` — current epoch (1-indexed)
    * ``step``  — global batch step counter
    * ``train_metrics`` — dict of latest training metrics
    * ``val_metrics``   — dict of latest validation metrics (may be empty)
    * ``best_metric``   — best monitored metric value seen so far
    * ``best_epoch``    — epoch where best was achieved
    * ``stop_training`` — set to ``True`` to request early stop
    """

    def on_train_begin(self, state: TrainingState) -> None:
        """Called once before the first epoch."""

    def on_train_end(self, state: TrainingState) -> None:
        """Called once after training completes (or is stopped early)."""

    def on_epoch_begin(self, state: TrainingState) -> None:
        """Called at the start of each epoch, before batches are processed."""

    def on_epoch_end(self, state: TrainingState) -> None:
        """Called at the end of each epoch, after validation."""

    def on_batch_begin(self, state: TrainingState) -> None:
        """Called before each mini-batch."""

    def on_batch_end(self, state: TrainingState) -> None:
        """Called after each mini-batch."""


# ---------------------------------------------------------------------------
# Callback list (internal composite)
# ---------------------------------------------------------------------------


class CallbackList:
    """Dispatches hook calls to a list of :class:`Callback` objects."""

    def __init__(self, callbacks: list[Callback]) -> None:
        self.callbacks = list(callbacks)

    def on_train_begin(self, state: TrainingState) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: TrainingState) -> None:
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_epoch_begin(self, state: TrainingState) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(state)

    def on_epoch_end(self, state: TrainingState) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(state)

    def on_batch_begin(self, state: TrainingState) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(state)

    def on_batch_end(self, state: TrainingState) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(state)


# ---------------------------------------------------------------------------
# Built-in callbacks
# ---------------------------------------------------------------------------


class CheckpointCallback(Callback):
    """Save model checkpoints during training.

    Keeps the *save_top_k* best checkpoints (ranked by *monitor* metric)
    and optionally always writes a ``last.pt`` file.

    Parameters
    ----------
    wrapper : TrainableModelWrapper
        The model wrapper whose :meth:`save_checkpoint` is called.
    dirpath : str or Path
        Directory for checkpoint files.
    monitor : str
        Metric key to watch (e.g. ``"val_loss"``).
    mode : {"min", "max"}
        Whether lower (``"min"``) or higher (``"max"``) values are better.
    save_top_k : int
        Maximum number of best checkpoints to keep (-1 = keep all).
    save_last : bool
        Write ``last.pt`` after every epoch regardless of metric value.
    verbose : bool
        Log checkpoint events.

    Example
    -------
    ::

        ckpt_cb = CheckpointCallback(
            wrapper=wrapper,
            dirpath="runs/my_run/checkpoints",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        )
        trainer = Trainer(..., callbacks=[ckpt_cb])
    """

    def __init__(
        self,
        wrapper: Any,
        dirpath: str | Path = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        verbose: bool = True,
    ) -> None:
        self.wrapper = wrapper
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose

        self._is_better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
        self._top_k: list[tuple[float, Path]] = []  # (metric, path)

    def on_train_begin(self, state: TrainingState) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, state: TrainingState) -> None:
        metrics = {**state.train_metrics, **state.val_metrics}
        value = metrics.get(self.monitor)
        if value is None:
            return

        # Always write last
        if self.save_last:
            last_path = self.dirpath / "last.pt"
            self.wrapper.save_checkpoint(last_path)

        # Check if this is a top-k checkpoint
        ckpt_name = f"epoch{state.epoch:04d}_{self.monitor}{value:.4f}.pt"
        ckpt_path = self.dirpath / ckpt_name

        should_save = (
            self.save_top_k < 0
            or len(self._top_k) < self.save_top_k
            or (self._top_k and self._is_better(value, self._top_k[-1][0]))
        )

        if should_save:
            self.wrapper.save_checkpoint(ckpt_path)
            self._top_k.append((value, ckpt_path))
            # Sort: best first
            self._top_k.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

            if self.save_top_k > 0 and len(self._top_k) > self.save_top_k:
                _, worst_path = self._top_k.pop()
                if worst_path.exists():
                    worst_path.unlink()

            if self.verbose:
                logger.info(
                    "Checkpoint saved: {} | {}={:.6f}",
                    ckpt_path.name,
                    self.monitor,
                    value,
                )


class EarlyStoppingCallback(Callback):
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    monitor : str
        Metric key to watch (e.g. ``"val_loss"``).
    mode : {"min", "max"}
    patience : int
        Number of epochs without improvement before stopping.
    min_delta : float
        Minimum absolute change that counts as an improvement.
    verbose : bool
        Log early-stopping events.

    Example
    -------
    ::

        es_cb = EarlyStoppingCallback(monitor="val_loss", patience=10)
        trainer = Trainer(..., callbacks=[es_cb])
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 1e-4,
        verbose: bool = True,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._wait: int = 0
        self._is_better = (
            (lambda new, best: new < best - min_delta)
            if mode == "min"
            else (lambda new, best: new > best + min_delta)
        )

    def on_epoch_end(self, state: TrainingState) -> None:
        metrics = {**state.train_metrics, **state.val_metrics}
        value = metrics.get(self.monitor)
        if value is None:
            return

        if self._is_better(value, self._best):
            self._best = value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                state.stop_training = True
                if self.verbose:
                    logger.info(
                        "Early stopping triggered after {} epochs without improvement "
                        "in '{}' (best={:.6f}, patience={}).",
                        self._wait,
                        self.monitor,
                        self._best,
                        self.patience,
                    )


class MetricLoggerCallback(Callback):
    """Append per-epoch metrics to a JSONL file for post-hoc analysis.

    Each line is a JSON object::

        {"epoch": 1, "step": 312, "train_loss": 0.045, "val_loss": 0.038, ...}

    Parameters
    ----------
    filepath : str or Path
        Destination ``.jsonl`` file.  Created (or appended) automatically.
    """

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)

    def on_train_begin(self, state: TrainingState) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, state: TrainingState) -> None:
        record: dict[str, Any] = {
            "epoch": state.epoch,
            "step": state.step,
        }
        record.update({f"train_{k}": v for k, v in state.train_metrics.items()})
        record.update({f"val_{k}": v for k, v in state.val_metrics.items()})

        with self.filepath.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


class LossPlotCallback(Callback):
    """Write a loss-curve PNG at the end of training.

    The callback reads the epoch-level ``state.metric_history`` generated by
    :class:`~facet.training.trainer.Trainer`. It is intentionally optional and
    isolated so training still works in environments without matplotlib.
    """

    def __init__(
        self,
        filepath: str | Path,
        *,
        train_key: str = "loss",
        val_key: str = "val_loss",
    ) -> None:
        self.filepath = Path(filepath)
        self.train_key = train_key
        self.val_key = val_key

    def on_train_begin(self, state: TrainingState) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_train_end(self, state: TrainingState) -> None:
        train_loss = state.metric_history.get(self.train_key, [])
        val_loss = state.metric_history.get(self.val_key, [])
        if not train_loss and not val_loss:
            return

        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            logger.warning("Skipping loss plot because matplotlib is not available: {}", exc)
            return

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if train_loss:
            epochs = range(1, len(train_loss) + 1)
            ax.plot(epochs, train_loss, marker="o", linewidth=1.5, label="train loss")
        if val_loss:
            epochs = range(1, len(val_loss) + 1)
            ax.plot(epochs, val_loss, marker="o", linewidth=1.5, label="validation loss")

        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()
        if _should_use_log_scale(train_loss, val_loss):
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(self.filepath, dpi=160)
        plt.close(fig)
        logger.info("Loss plot saved: {}", self.filepath)


def _should_use_log_scale(*series: list[float]) -> bool:
    values = [float(value) for values in series for value in values if value > 0]
    if len(values) < 2:
        return False
    return max(values) / min(values) >= 100.0


class WandbCallback(Callback):
    """Log metrics to Weights & Biases (optional dependency).

    Requires ``wandb`` to be installed::

        pip install wandb

    Parameters
    ----------
    project : str
        W&B project name.
    run_name : str or None
        Run name shown in the W&B dashboard.
    config : dict or None
        Hyperparameter dict logged to W&B.
    tags : list[str] or None
        Run tags.

    Example
    -------
    ::

        wb_cb = WandbCallback(project="facetpy-dl", run_name="eegdfus-run1")
        trainer = Trainer(..., callbacks=[wb_cb])
    """

    def __init__(
        self,
        project: str = "facetpy",
        run_name: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.tags = tags or []
        self._run: Any = None

    def on_train_begin(self, state: TrainingState) -> None:
        try:
            import wandb  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "wandb is required for WandbCallback. Install with: pip install wandb"
            ) from exc

        self._run = wandb.init(
            project=self.project,
            name=self.run_name or state.run_name,
            config=self.config,
            tags=self.tags,
            reinit=True,
        )

    def on_epoch_end(self, state: TrainingState) -> None:
        if self._run is None:
            return
        log_dict: dict[str, Any] = {"epoch": state.epoch}
        log_dict.update({f"train/{k}": v for k, v in state.train_metrics.items()})
        log_dict.update({f"val/{k}": v for k, v in state.val_metrics.items()})
        self._run.log(log_dict, step=state.step)

    def on_train_end(self, state: TrainingState) -> None:
        if self._run is not None:
            self._run.finish()
