"""Training callbacks — checkpoint saving, early stopping, and extensibility hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from .trainer import TrainingState
    from .wrapper import TrainableModelWrapper


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
        # Skip epochs whose monitored value is missing or non-finite: never
        # checkpoint, rank, or early-stop on a NaN/inf metric.
        if value is None or not np.isfinite(value):
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
        # Skip epochs whose monitored value is missing or non-finite: never
        # checkpoint, rank, or early-stop on a NaN/inf metric.
        if value is None or not np.isfinite(value):
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


class SavePredictionSamplesCallback(Callback):
    """Snapshot a fixed batch of validation predictions every N epochs.

    Picks ``n_samples`` validation indices once at ``on_train_begin`` (using
    a deterministic seed) and re-uses them every snapshot, so the resulting
    plots are directly comparable across epochs. Each snapshot writes:

    * ``<output_dir>/epoch_NNNN.npz`` — raw arrays ``indices``, ``noisy``,
      ``target``, ``prediction`` for downstream analysis.
    * ``<output_dir>/epoch_NNNN.png`` — one row per sample with target and
      prediction overlaid on the same axes.

    Parameters
    ----------
    wrapper : TrainableModelWrapper
        Model wrapper; must implement :meth:`predict_batch`.
    val_dataset : dataset-like or None
        Validation dataset. When ``None`` or empty, the callback is a no-op.
    output_dir : str or Path
        Destination directory (created automatically).
    n_samples : int
        Number of validation samples to snapshot.
    every_n_epochs : int
        Snapshot cadence. ``1`` means after every epoch.
    seed : int
        Seed used to draw the fixed sample indices once at train start.
    verbose : bool
        Log snapshot events.
    """

    def __init__(
        self,
        wrapper: TrainableModelWrapper,
        val_dataset: Any,
        output_dir: str | Path,
        *,
        n_samples: int = 4,
        every_n_epochs: int = 5,
        seed: int = 7,
        verbose: bool = True,
    ) -> None:
        self.wrapper = wrapper
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.n_samples = int(n_samples)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.seed = int(seed)
        self.verbose = verbose
        self._indices: list[int] | None = None

    def on_train_begin(self, state: TrainingState) -> None:
        if self.val_dataset is None or len(self.val_dataset) == 0:
            logger.warning(
                "SavePredictionSamplesCallback: validation dataset is empty; "
                "snapshots disabled."
            )
            self._indices = None
            return
        n = len(self.val_dataset)
        k = min(self.n_samples, n)
        rng = np.random.default_rng(self.seed)
        self._indices = sorted(int(i) for i in rng.choice(n, size=k, replace=False))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            logger.info(
                "SavePredictionSamplesCallback: snapshotting {} val sample(s) "
                "every {} epoch(s) to {}",
                k,
                self.every_n_epochs,
                self.output_dir,
            )

    def on_epoch_end(self, state: TrainingState) -> None:
        if not self._indices:
            return
        if state.epoch % self.every_n_epochs != 0:
            return

        noisy_list, target_list = zip(
            *(self.val_dataset[idx] for idx in self._indices), strict=False
        )
        noisy = np.stack(noisy_list, axis=0)
        target = np.stack(target_list, axis=0)

        try:
            prediction = self.wrapper.predict_batch(noisy)
        except NotImplementedError:
            logger.warning(
                "SavePredictionSamplesCallback: wrapper {} does not implement "
                "predict_batch(); disabling further snapshots.",
                type(self.wrapper).__name__,
            )
            self._indices = None
            return

        prediction = np.asarray(prediction, dtype=np.float32)

        self._write_npz(state.epoch, noisy, target, prediction)
        self._write_plot(state.epoch, target, prediction)
        if self.verbose:
            logger.info(
                "SavePredictionSamplesCallback: wrote epoch {:04d} snapshot.",
                state.epoch,
            )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _write_npz(
        self,
        epoch: int,
        noisy: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
    ) -> None:
        path = self.output_dir / f"epoch_{epoch:04d}.npz"
        np.savez(
            path,
            indices=np.asarray(self._indices, dtype=np.int64),
            noisy=noisy.astype(np.float32, copy=False),
            target=target.astype(np.float32, copy=False),
            prediction=prediction,
        )

    def _write_plot(
        self,
        epoch: int,
        target: np.ndarray,
        prediction: np.ndarray,
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            logger.warning(
                "Skipping prediction snapshot plot because matplotlib is "
                "not available: {}",
                exc,
            )
            return

        n = target.shape[0]
        fig, axes = plt.subplots(n, 1, figsize=(12, 2.4 * n), squeeze=False)
        for row, sample_idx in enumerate(self._indices or []):
            ax = axes[row, 0]
            tgt = _to_1d(target[row])
            pred = _to_1d(prediction[row])
            x = np.arange(min(tgt.size, pred.size))
            ax.plot(
                x,
                tgt[: x.size] * 1e6,
                color="black",
                linewidth=1.0,
                alpha=0.75,
                label="target",
            )
            ax.plot(
                x,
                pred[: x.size] * 1e6,
                color="#dc2626",
                linewidth=1.0,
                alpha=0.75,
                label="prediction",
            )
            ax.set_ylabel("µV")
            ax.set_title(f"val sample {sample_idx}")
            ax.grid(alpha=0.25)
            if row == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1, 0].set_xlabel("sample")
        fig.suptitle(f"Predictions @ epoch {epoch}")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"epoch_{epoch:04d}.png", dpi=140)
        plt.close(fig)


def _to_1d(arr: np.ndarray) -> np.ndarray:
    """Reduce an arbitrary-rank prediction/target to a 1D view for plotting.

    Squeezes singleton axes; for multi-channel arrays, returns the first
    channel. Always returns a contiguous 1D float array.
    """
    a = np.asarray(arr)
    while a.ndim > 1 and a.shape[0] == 1:
        a = a.squeeze(0)
    if a.ndim > 1:
        a = a[0]
    return np.ascontiguousarray(a, dtype=np.float32)
