"""Trainer — framework-agnostic training loop with Rich Live dashboard."""

from __future__ import annotations

import datetime
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LossPlotCallback,
    MetricLoggerCallback,
)
from .config import TrainingConfig
from .wrapper import TrainableModelWrapper

# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 10) -> str:
    """Render a list of floats as a unicode sparkline string."""
    if not values:
        return " " * width
    tail = values[-width:]
    lo, hi = min(tail), max(tail)
    span = hi - lo or 1.0
    bars = [_SPARK_CHARS[int((v - lo) / span * (len(_SPARK_CHARS) - 1))] for v in tail]
    return "".join(bars)


def _trend_arrow(values: list[float]) -> str:
    if len(values) < 2:
        return " "
    delta = values[-1] - values[-2]
    if abs(delta) < 1e-9:
        return "→"
    return "↓" if delta < 0 else "↑"


@dataclass
class TrainingState:
    """Mutable state object shared between Trainer and Callbacks.

    Callbacks may read any field.  Callbacks may set ``stop_training``
    to ``True`` to request graceful termination.
    """

    # Identity
    run_name: str = ""

    # Progress
    epoch: int = 0
    max_epochs: int = 0
    step: int = 0                      # global batch step counter
    n_train_batches: int = 0
    n_val_batches: int = 0

    # Latest metrics (updated after each epoch)
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)

    # Per-epoch history (for sparklines)
    metric_history: dict[str, list[float]] = field(default_factory=dict)

    # Best checkpoint tracking
    best_metric: float = float("nan")
    best_epoch: int = 0

    # Control
    stop_training: bool = False

    # Timing
    train_start_time: float = field(default_factory=time.monotonic)
    epoch_start_time: float = field(default_factory=time.monotonic)


@dataclass
class TrainingResult:
    """Returned by :meth:`Trainer.fit` when training completes.

    Attributes
    ----------
    success : bool
        ``True`` if training finished without exceptions.
    total_epochs : int
        Number of epochs actually completed.
    best_epoch : int
        Epoch where the monitored metric was best.
    best_metric : float
        Best value of the monitored metric.
    metric_history : dict[str, list[float]]
        Per-epoch metric values for all logged metrics.
    elapsed_seconds : float
        Wall-clock time consumed by :meth:`fit`.
    run_dir : Path or None
        Output directory for this run (checkpoints, logs).
    """

    success: bool
    total_epochs: int
    best_epoch: int
    best_metric: float
    metric_history: dict[str, list[float]]
    elapsed_seconds: float
    run_dir: Path | None = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Framework-agnostic training loop with a Rich Live terminal dashboard.

    The Trainer drives the epoch / batch loop and delegates all
    framework-specific work to a :class:`~facet.training.wrapper.TrainableModelWrapper`.
    Callbacks are invoked at each hook point; the Rich Live display
    updates automatically.

    Terminal dashboard layout::

        ╭─ FACETpy Training ─ EEGDfus ─ run_20260409_143022 ──────────────╮
        │  Device: cuda:0  │  Epoch 12 / 50                                │
        │  [██████████████████░░░░░░░░░]  Batch 45/63  71%  ETA 18m 32s   │
        ├──────────────────────────────────────────────────────────────────┤
        │  Metric              Value       Trend    Last 10 epochs         │
        │  train_loss          0.01243     ↓        ▇▇▆▅▅▄▄▃▃▃            │
        │  val_loss            0.01891     →        ▇▇▇▆▆▅▅▅▅▅  ★ best   │
        ╰──────────────────────────────────────────────────────────────────╯

    Parameters
    ----------
    wrapper : TrainableModelWrapper
        The model to train.
    train_dataset : dataset-like
        Any object with ``__len__`` and ``__getitem__(idx)`` returning
        ``(noisy, target)`` numpy arrays.
    val_dataset : dataset-like or None
        Validation dataset.  When ``None``, no validation is run.
    config : TrainingConfig
        Full training configuration.
    callbacks : list of Callback, optional
        Additional callbacks.  :class:`CheckpointCallback` and
        :class:`MetricLoggerCallback` are added automatically based on
        *config* unless provided explicitly.
    seed : int or None
        Random seed for batch shuffling.  Overrides ``config.seed``.

    Example
    -------
    ::

        from facet.training import (
            EEGArtifactDataset, PyTorchModelWrapper, Trainer, TrainingConfig,
        )

        dataset = EEGArtifactDataset(context, chunk_size=1250)
        train_ds, val_ds = dataset.train_val_split()

        wrapper = PyTorchModelWrapper(model=my_model, loss_fn=nn.MSELoss())

        config = TrainingConfig(model_name="MyModel", max_epochs=50, batch_size=16)

        result = Trainer(wrapper, train_ds, val_ds, config).fit()
        print(f"Best val_loss: {result.best_metric:.4f} @ epoch {result.best_epoch}")
    """

    def __init__(
        self,
        wrapper: TrainableModelWrapper,
        train_dataset: Any,
        val_dataset: Any | None = None,
        config: TrainingConfig | None = None,
        callbacks: list[Callback] | None = None,
        seed: int | None = None,
    ) -> None:
        self.wrapper = wrapper
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        self._seed = seed if seed is not None else self.config.seed
        self._extra_callbacks: list[Callback] = list(callbacks or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> TrainingResult:
        """Run the full training loop and return a :class:`TrainingResult`.

        Returns
        -------
        TrainingResult
        """
        run_name = self.config.run_name or _auto_run_name(self.config.model_name)
        run_dir = Path(self.config.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Persist config
        self.config.save_json(run_dir / "config.json")

        state = TrainingState(
            run_name=run_name,
            max_epochs=self.config.max_epochs,
        )

        callbacks = self._build_callbacks(run_dir)

        t0 = time.monotonic()
        success = False
        try:
            if self.config.logging.rich_live:
                self._fit_with_live(state, callbacks, run_dir)
            else:
                self._fit_plain(state, callbacks)
            success = True
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        except Exception as exc:
            logger.exception("Training failed: {}", exc)
            raise
        finally:
            callbacks.on_train_end(state)

        elapsed = time.monotonic() - t0
        return TrainingResult(
            success=success,
            total_epochs=state.epoch,
            best_epoch=state.best_epoch,
            best_metric=state.best_metric,
            metric_history=dict(state.metric_history),
            elapsed_seconds=elapsed,
            run_dir=run_dir,
        )

    # ------------------------------------------------------------------
    # Dashboard display
    # ------------------------------------------------------------------

    def _fit_with_live(
        self, state: TrainingState, callbacks: CallbackList, run_dir: Path
    ) -> None:
        console = Console()
        progress = self._make_progress()
        epoch_task = progress.add_task("Epoch", total=self.config.max_epochs)
        batch_task = progress.add_task("Batch", total=1, visible=False)

        def _make_layout() -> Layout:
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="progress", size=3),
                Layout(name="metrics"),
            )
            return layout

        def _render_header() -> Panel:
            title = Text()
            title.append("FACETpy Training", style="bold cyan")
            title.append(f"  ·  {self.config.model_name}", style="bold white")
            title.append(f"  ·  {state.run_name}", style="dim")
            subtitle = Text()
            subtitle.append(f"Device: {self.wrapper.device_info}", style="green")
            subtitle.append(f"  │  Epoch {state.epoch} / {self.config.max_epochs}", style="white")
            return Panel(
                subtitle,
                title=title,
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            )

        def _render_metrics() -> Panel:
            table = Table(
                box=None,
                show_header=True,
                expand=True,
                show_edge=False,
                padding=(0, 2),
            )
            table.add_column("Metric", style="bold", ratio=3)
            table.add_column("Value", ratio=2, justify="right")
            table.add_column("", ratio=1, justify="center")   # trend arrow
            table.add_column("Last 10 epochs", ratio=4)
            table.add_column("", ratio=2, style="dim italic")  # annotation

            all_metrics = {**state.train_metrics, **{f"val_{k}": v for k, v in state.val_metrics.items()}}
            monitor = self.config.checkpoint.monitor

            for name, value in sorted(all_metrics.items()):
                history = state.metric_history.get(name, [])
                spark = _sparkline(history)
                arrow = _trend_arrow(history)

                is_best = (name == monitor and state.epoch == state.best_epoch)
                annotation = "★ best" if is_best else ""
                color = "green" if arrow == "↓" else ("red" if arrow == "↑" else "white")
                table.add_row(
                    name,
                    f"{value:.6f}",
                    f"[{color}]{arrow}[/]",
                    spark,
                    annotation,
                )

            return Panel(
                table,
                title="[bold white] Metrics [/]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
            )

        layout = _make_layout()

        with Live(layout, console=console, refresh_per_second=4, transient=False):
            callbacks.on_train_begin(state)

            for epoch in range(1, self.config.max_epochs + 1):
                state.epoch = epoch
                state.epoch_start_time = time.monotonic()
                callbacks.on_epoch_begin(state)

                # ---- train ----
                train_metrics = self._run_epoch(
                    state, "train", callbacks, progress, batch_task
                )
                state.train_metrics = train_metrics

                # ---- validate ----
                if self.val_dataset is not None and epoch % self.config.logging.val_every_n_epochs == 0:
                    val_metrics = self._run_epoch(
                        state, "val", callbacks, progress, batch_task
                    )
                    state.val_metrics = val_metrics
                else:
                    state.val_metrics = {}

                # Update history
                for k, v in {**train_metrics, **state.val_metrics}.items():
                    state.metric_history.setdefault(k, []).append(v)

                # Track best
                self._update_best(state)

                callbacks.on_epoch_end(state)
                self._step_scheduler()
                progress.update(epoch_task, advance=1)

                # Refresh layout
                layout["header"].update(_render_header())
                layout["progress"].update(progress)
                layout["metrics"].update(_render_metrics())

                if state.stop_training:
                    break

    def _fit_plain(self, state: TrainingState, callbacks: CallbackList) -> None:
        """Fallback training loop without Rich Live (non-TTY / CI)."""
        callbacks.on_train_begin(state)

        for epoch in range(1, self.config.max_epochs + 1):
            state.epoch = epoch
            callbacks.on_epoch_begin(state)

            train_metrics = self._run_epoch(state, "train", callbacks)
            state.train_metrics = train_metrics

            if self.val_dataset is not None and epoch % self.config.logging.val_every_n_epochs == 0:
                val_metrics = self._run_epoch(state, "val", callbacks)
                state.val_metrics = val_metrics
            else:
                state.val_metrics = {}

            for k, v in {**train_metrics, **state.val_metrics}.items():
                state.metric_history.setdefault(k, []).append(v)

            self._update_best(state)
            callbacks.on_epoch_end(state)
            self._step_scheduler()

            # Log to console
            metric_str = "  ".join(
                f"{k}={v:.4f}"
                for k, v in {**train_metrics, **state.val_metrics}.items()
            )
            logger.info("Epoch {:03d}/{:03d}  {}", epoch, self.config.max_epochs, metric_str)

            if state.stop_training:
                break

    # ------------------------------------------------------------------
    # Epoch / batch loop (shared)
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        state: TrainingState,
        mode: str,
        callbacks: CallbackList,
        progress: Progress | None = None,
        batch_task: Any = None,
    ) -> dict[str, float]:
        dataset = self.train_dataset if mode == "train" else self.val_dataset
        n = len(dataset)
        batch_size = self.config.batch_size

        indices = np.arange(n)
        if mode == "train":
            rng = np.random.default_rng(self._seed + state.epoch)
            rng.shuffle(indices)

        n_batches = max(1, math.ceil(n / batch_size))
        if progress is not None and batch_task is not None:
            progress.reset(batch_task, total=n_batches, visible=True, description=mode)

        accumulated: dict[str, list[float]] = {}

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            noisy_list, target_list = zip(
                *[dataset[int(i)] for i in batch_indices], strict=False
            )
            noisy = np.stack(noisy_list, axis=0)    # (B, C, T)
            target = np.stack(target_list, axis=0)  # (B, C, T)

            callbacks.on_batch_begin(state)

            if mode == "train":
                metrics = self.wrapper.train_step(noisy, target)
                state.step += 1
            else:
                metrics = self.wrapper.eval_step(noisy, target)

            callbacks.on_batch_end(state)

            for k, v in metrics.items():
                accumulated.setdefault(k, []).append(v)

            if progress is not None and batch_task is not None:
                progress.advance(batch_task)

        if progress is not None and batch_task is not None:
            progress.update(batch_task, visible=False)

        return {k: float(np.mean(vs)) for k, vs in accumulated.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_best(self, state: TrainingState) -> None:
        monitor = self.config.checkpoint.monitor
        mode = self.config.checkpoint.mode
        all_metrics = {**state.train_metrics, **state.val_metrics}
        value = all_metrics.get(monitor)
        if value is None:
            return
        if math.isnan(state.best_metric):
            state.best_metric = value
            state.best_epoch = state.epoch
            return
        improved = value < state.best_metric if mode == "min" else value > state.best_metric
        if improved:
            state.best_metric = value
            state.best_epoch = state.epoch

    def _build_callbacks(self, run_dir: Path) -> CallbackList:
        callbacks = list(self._extra_callbacks)

        # Auto-add CheckpointCallback if not provided
        has_ckpt = any(isinstance(cb, CheckpointCallback) for cb in callbacks)
        if not has_ckpt:
            callbacks.append(
                CheckpointCallback(
                    wrapper=self.wrapper,
                    dirpath=run_dir / self.config.checkpoint.dirpath,
                    monitor=self.config.checkpoint.monitor,
                    mode=self.config.checkpoint.mode,
                    save_top_k=self.config.checkpoint.save_top_k,
                    save_last=self.config.checkpoint.save_last,
                )
            )

        # Auto-add MetricLoggerCallback if configured
        has_logger = any(isinstance(cb, MetricLoggerCallback) for cb in callbacks)
        if not has_logger and self.config.logging.log_file:
            callbacks.append(
                MetricLoggerCallback(filepath=run_dir / self.config.logging.log_file)
            )

        has_loss_plot = any(isinstance(cb, LossPlotCallback) for cb in callbacks)
        if not has_loss_plot and self.config.logging.loss_plot_file:
            callbacks.append(
                LossPlotCallback(filepath=run_dir / self.config.logging.loss_plot_file)
            )

        has_early_stopping = any(
            isinstance(cb, EarlyStoppingCallback) for cb in callbacks
        )
        if not has_early_stopping and self.config.early_stopping is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=self.config.early_stopping.monitor,
                    mode=self.config.early_stopping.mode,
                    patience=self.config.early_stopping.patience,
                    min_delta=self.config.early_stopping.min_delta,
                )
            )

        return CallbackList(callbacks)

    def _step_scheduler(self) -> None:
        scheduler_step = getattr(self.wrapper, "scheduler_step", None)
        if callable(scheduler_step):
            scheduler_step()

    @staticmethod
    def _make_progress() -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=False,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _auto_run_name(model_name: str) -> str:
    """Generate a timestamped run name."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = model_name.lower().replace(" ", "_")
    return f"{slug}_{ts}"
