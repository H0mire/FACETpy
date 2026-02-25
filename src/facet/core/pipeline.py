"""
Pipeline Module

This module defines the Pipeline class for executing sequences of processors.

Author: FACETpy Team
Date: 2025-01-12
"""

from collections.abc import Callable
from typing import Any

from loguru import logger

from ..console import get_console
from ..console.progress import set_current_step_index
from .channel_sequential import ChannelSequentialExecutor
from .context import ProcessingContext
from .parallel import ParallelExecutor
from .processor import Processor


class PipelineError(Exception):
    """Base exception for pipeline-related errors."""

    pass


class PipelineResult:
    """
    Result of pipeline execution.

    Contains the final context and metadata about execution.
    """

    def __init__(
        self,
        context: ProcessingContext | None,
        success: bool = True,
        error: Exception | None = None,
        execution_time: float = 0.0,
        failed_processor: str | None = None,
        failed_processor_index: int | None = None,
    ):
        """
        Initialize pipeline result.

        Args:
            context: Final processing context
            success: Whether pipeline completed successfully
            error: Exception if pipeline failed
            execution_time: Total execution time in seconds
        """
        self.context = context
        self.success = success
        self.error = error
        self.execution_time = execution_time
        self.failed_processor = failed_processor
        self.failed_processor_index = failed_processor_index

    def get_context(self) -> ProcessingContext:
        """Get final processing context."""
        return self.context

    def get_raw(self):
        """Get final raw data (convenience method)."""
        return self.context.get_raw()

    def get_history(self):
        """Get processing history."""
        return self.context.get_history()

    def was_successful(self) -> bool:
        """Check if pipeline succeeded."""
        return self.success

    @property
    def metrics(self) -> dict[str, Any]:
        """
        Shortcut to evaluation metrics stored in context.

        Returns the ``metrics`` dict from ``context.metadata.custom``, or an
        empty dict if no metrics have been calculated yet.

        Example::

            result = pipeline.run()
            print(result.metrics['snr'])
        """
        if self.context is None:
            return {}
        return self.context.metadata.custom.get("metrics", {})

    @property
    def metrics_df(self):
        """
        Return scalar evaluation metrics as a ``pandas.Series``.

        Nested dicts (e.g. ``fft_allen``) are flattened with ``_`` separators.
        Returns ``None`` if pandas is not available.

        Example::

            result = pipeline.run()
            print(result.metrics_df)
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        flat: dict[str, Any] = {}
        for k, v in self.metrics.items():
            if isinstance(v, (int, float)):
                flat[k] = v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        flat[f"{k}_{sub_k}"] = sub_v
        return pd.Series(flat, name=self.context.metadata.custom.get("pipeline_name", "metrics"))

    def metric(self, name: str, default=None):
        """
        Return a single evaluation metric by name.

        Shortcut for ``result.metrics.get(name, default)`` that avoids having
        to remember the dict key and provides a clean default.

        Args:
            name: Metric name (e.g. ``'snr'``, ``'rms_ratio'``).
            default: Value returned when the metric is absent.

        Example::

            snr = result.metric('snr')
            print(f"SNR = {snr:.2f} dB")
        """
        return self.metrics.get(name, default)

    def print_metrics(self) -> None:
        """
        Print a formatted table of all evaluation metrics.

        Uses *rich* for colour and alignment when available.

        Example::

            result = pipeline.run()
            result.print_metrics()
        """
        import numpy as np
        from rich import box
        from rich.console import Console as RichConsole
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        metrics = self.metrics
        if not metrics:
            print("No metrics available — did you add evaluation processors?")
            return

        con = get_console().get_rich_console() or RichConsole(highlight=False)
        table = Table(
            box=None,
            show_header=True,
            padding=(0, 2),
            expand=True,
            show_edge=False,
        )
        table.add_column("Metric", style="bold", ratio=3)
        table.add_column("Value", style="white", ratio=2, justify="left")
        table.add_column("", style="dim italic", ratio=1)

        def _section(title: str) -> None:
            table.add_row("", "", "")
            table.add_row(Text(title, style="bold yellow underline"), "", "")

        def _fmt_per_channel(val: list) -> str:
            arr = np.asarray(val, dtype=float)
            return f"mean {arr.mean():.3g}  ± {arr.std():.3g}  [dim](min {arr.min():.3g} – max {arr.max():.3g})[/]"

        def _color_snr(v: float) -> str:
            return "green" if v > 10 else ("yellow" if v > 3 else "red")

        def _color_ratio(v: float) -> str:
            return "green" if abs(v - 1.0) < 0.1 else ("yellow" if abs(v - 1.0) < 0.3 else "red")

        # --- Core scalar metrics ---
        core_keys = ("snr", "rms_ratio", "rms_residual", "median_artifact", "legacy_snr")
        if any(k in metrics for k in core_keys):
            _section("Core Metrics")
            if "snr" in metrics:
                snr = metrics["snr"]
                c = _color_snr(snr)
                table.add_row("SNR (Signal-to-Noise Ratio)", f"[{c}]{snr:.2f}[/]", "")
            if "rms_ratio" in metrics:
                table.add_row("RMS Ratio (improvement)", f"{metrics['rms_ratio']:.2f}", "×")
            if "rms_residual" in metrics:
                r = metrics["rms_residual"]
                c = _color_ratio(r)
                table.add_row("RMS Residual Ratio", f"[{c}]{r:.2f}[/]", "target: 1.0")
            if "median_artifact" in metrics:
                table.add_row("Median Artifact Amplitude", f"{metrics['median_artifact']:.2e}", "")
                if "median_artifact_ratio" in metrics:
                    r = metrics["median_artifact_ratio"]
                    c = "green" if abs(r - 1.0) < 0.2 else ("yellow" if abs(r - 1.0) < 0.6 else "red")
                    table.add_row("Median Artifact Ratio", f"[{c}]{r:.2f}[/]", "target: 1.0")
            if "legacy_snr" in metrics:
                table.add_row("Legacy SNR", f"{metrics['legacy_snr']:.2f}", "")

        # --- Per-channel breakdowns ---
        per_ch = {k: v for k, v in metrics.items() if k.endswith("_per_channel") and isinstance(v, list)}
        if per_ch:
            _section("Per-Channel Summary  (mean ± std,  min – max)")
            for key, val in per_ch.items():
                label = key.replace("_per_channel", "").replace("_", " ").title()
                table.add_row(label, _fmt_per_channel(val), "")

        # --- FFT Allen ---
        if "fft_allen" in metrics:
            _section("FFT Allen — Spectral Diff to Reference")
            for band, val in metrics["fft_allen"].items():
                table.add_row(f"{band.capitalize()}", f"{val:.2f}%", "")

        # --- FFT Niazy ---
        if "fft_niazy" in metrics:
            _section("FFT Niazy — Power Ratio (Uncorr / Corr)")
            if "slice" in metrics["fft_niazy"]:
                harmonics = "  ".join(f"[cyan]{k}[/]: {v:.2f}" for k, v in metrics["fft_niazy"]["slice"].items())
                table.add_row("Slice Harmonics", harmonics, "dB")
            if "volume" in metrics["fft_niazy"]:
                harmonics = "  ".join(f"[cyan]{k}[/]: {v:.2f}" for k, v in metrics["fft_niazy"]["volume"].items())
                table.add_row("Volume Harmonics", harmonics, "dB")

        # --- Other unknown keys ---
        known = (
            set(core_keys)
            | set(per_ch)
            | {"median_artifact_ratio", "median_artifact_reference", "fft_allen", "fft_niazy"}
        )
        extras = {k: v for k, v in metrics.items() if k not in known}
        if extras:
            _section("Other")
            for key, val in extras.items():
                label = key.replace("_", " ").title()
                if isinstance(val, float):
                    formatted = f"{val:.4g}"
                elif isinstance(val, dict):
                    formatted = "  ".join(
                        f"{k}: {v:.3g}" if isinstance(v, float) else f"{k}: {v}" for k, v in val.items()
                    )
                elif isinstance(val, list):
                    formatted = _fmt_per_channel(val) if val and isinstance(val[0], (int, float)) else str(val)
                else:
                    formatted = str(val)
                table.add_row(label, formatted, "")

        con.print()
        con.print(
            Panel(
                table,
                title="[bold white] Evaluation Metrics Report [/]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def print_summary(self) -> None:
        """
        Print a one-line summary of the pipeline result.

        Shows success/failure, execution time, and any key metrics (SNR, RMS
        ratio, RMS residual) that were calculated.

        Example::

            result = pipeline.run()
            result.print_summary()
        """
        from rich.console import Console as RichConsole

        con = get_console().get_rich_console() or RichConsole()
        if self.success:
            parts = [f"[green]Done[/green] in {self.execution_time:.2f}s"]
            for name in ("snr", "rms_ratio", "rms_residual", "median_artifact"):
                val = self.metrics.get(name)
                if val is not None:
                    parts.append(f"{name}={val:.3g}")
            con.print("  ".join(parts))
        else:
            con.print(f"[red]Failed[/red] after {self.execution_time:.2f}s — {self.error}")

    def plot(self, **kwargs):
        """
        Plot the corrected data using ``RawPlotter`` defaults.

        Accepts any keyword arguments supported by ``RawPlotter``.

        Example::

            result = pipeline.run()
            result.plot(channel="Fp1", start=5.0, duration=10.0)
        """
        from ..evaluation import RawPlotter

        plotter = RawPlotter(**kwargs)
        plotter.execute(self.context)

    def release_raw(self) -> None:
        """
        Release the Raw data held by the context to free memory.

        After calling this, ``get_raw()`` and ``plot()`` will no longer work,
        but :attr:`metrics` and :attr:`execution_time` remain accessible.
        Useful when running batch jobs where you only need summary statistics.
        """
        if self.context is not None:
            self.context._raw = None
            self.context._raw_original = None

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"PipelineResult({status}, time={self.execution_time:.2f}s)"


class BatchResult:
    """
    Result of :meth:`Pipeline.map` — a list of :class:`PipelineResult` objects
    with built-in helpers for quick inspection.

    It behaves like a regular list (iteration, indexing, ``len``), so existing
    code that iterates over the return value of ``Pipeline.map()`` continues to
    work without changes.

    Example::

        results = pipeline.map(files, loader_factory=lambda p: Loader(p))
        results.print_summary()
        df = results.summary_df
    """

    def __init__(
        self,
        results: list["PipelineResult"],
        labels: list[str] | None = None,
    ):
        self._results = results
        self._labels = labels or [f"input_{i}" for i in range(len(results))]

    # ------------------------------------------------------------------
    # List-like interface
    # ------------------------------------------------------------------

    def __iter__(self):
        return iter(self._results)

    def __getitem__(self, index):
        return self._results[index]

    def __len__(self):
        return len(self._results)

    def __repr__(self):
        n_ok = sum(1 for r in self._results if r.success)
        return f"BatchResult({n_ok}/{len(self._results)} succeeded)"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """
        Print a formatted table with one row per input file.

        Columns include the file label, success/failure status, execution time,
        and any scalar metrics that were computed.

        Example::

            results = pipeline.map(files, loader_factory=...)
            results.print_summary()
        """
        from rich import box
        from rich.console import Console as RichConsole
        from rich.table import Table

        con = get_console().get_rich_console() or RichConsole(highlight=False)
        table = Table(
            title="Batch Results",
            show_header=True,
            header_style="bold cyan",
            box=box.SIMPLE_HEAVY,
            padding=(0, 1),
        )
        table.add_column("File", style="bold", no_wrap=True)
        table.add_column("Status", justify="left")
        table.add_column("Time", justify="left")

        metric_names: list[str] = []
        for r in self._results:
            for k, v in r.metrics.items():
                if k not in metric_names and isinstance(v, (int, float)):
                    metric_names.append(k)

        for m in metric_names:
            table.add_column(m, justify="left")

        for label, result in zip(self._labels, self._results, strict=False):
            status = "[green]OK[/green]" if result.success else "[red]FAIL[/red]"
            time_str = f"{result.execution_time:.2f}s"
            row: list[str] = [label, status, time_str]
            for m in metric_names:
                if result.success:
                    val = result.metrics.get(m)
                    row.append(f"{val:.3f}" if isinstance(val, float) else (str(val) if val is not None else "—"))
                else:
                    row.append("—")
            table.add_row(*row)

        con.print(table)

    @property
    def summary_df(self):
        """
        Return a ``pandas.DataFrame`` with one row per input.

        Columns: ``file``, ``success``, ``execution_time``, plus one column per
        scalar metric.  Returns ``None`` when *pandas* is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        rows = []
        for label, result in zip(self._labels, self._results, strict=False):
            row: dict[str, Any] = {
                "file": label,
                "success": result.success,
                "execution_time": result.execution_time,
            }
            if result.success and result.metrics_df is not None:
                row.update(result.metrics_df.to_dict())
            rows.append(row)
        return pd.DataFrame(rows)


class Pipeline:
    """
    Pipeline for executing sequences of processors.

    A pipeline orchestrates the execution of multiple processors in sequence,
    handles errors, provides progress tracking, and supports parallelization.

    Example::

        pipeline = Pipeline([
            Loader("data.edf"),
            HighPassFilter(freq=1.0),
            UpSample(factor=10),
            TriggerDetector(regex=r"\\btrigger\\b"),
            AASCorrection(),
            EDFExporter("output.edf")
        ])

        result = pipeline.run()
        if result.was_successful():
            print(f"Completed in {result.execution_time:.2f}s")

    Attributes:
        processors: List of processors to execute
        name: Optional pipeline name
    """

    def __init__(self, processors: list[Processor | Callable], name: str | None = None):
        """
        Initialize pipeline.

        Plain callables (``Callable[[ProcessingContext], ProcessingContext]``)
        are automatically wrapped in a :class:`~facet.core.LambdaProcessor` so
        they can be used as inline steps without ceremony::

            pipeline = Pipeline([
                Loader("data.edf"),
                HighPassFilter(1.0),
                lambda ctx: (print(ctx.get_sfreq()) or ctx),
                AASCorrection(),
            ])

        Args:
            processors: List of :class:`~facet.core.Processor` instances **or**
                plain callables to execute in order.
            name: Optional pipeline name (for logging)
        """
        self.processors = self._normalise_processors(processors)
        self.name = name or "Pipeline"

    @staticmethod
    def _normalise_processors(
        items: list[Processor | Callable],
        _index_offset: int = 0,
    ) -> list[Processor]:
        """
        Coerce each item to a :class:`Processor`.

        Plain callables are wrapped in a :class:`~facet.core.LambdaProcessor`.
        Anything else that is not a :class:`Processor` raises :exc:`TypeError`.
        """
        from .processor import LambdaProcessor

        result: list[Processor] = []
        for i, p in enumerate(items):
            if isinstance(p, Processor):
                result.append(p)
            elif callable(p):
                display_name = getattr(p, "__name__", None) or f"step_{_index_offset + i}"
                result.append(LambdaProcessor(name=display_name, func=p))
            else:
                raise TypeError(
                    f"Item at index {_index_offset + i} must be a Processor instance or a callable, got {type(p)}"
                )
        return result

    def _validate_pipeline(self) -> None:
        """No-op — validation now happens in _normalise_processors."""

    # ---------------------------------------------------------------------- #
    # Execution helpers                                                       #
    # ---------------------------------------------------------------------- #

    def _group_processors(
        self,
        parallel: bool,
        channel_sequential: bool,
    ) -> list[tuple[list[Processor], str]]:
        """
        Partition processors into execution groups.

        Returns a list of ``(processors, mode)`` tuples where *mode* is one of
        ``'channel_sequential'``, ``'parallel'``, or ``'serial'``.

        In channel_sequential mode consecutive processors with
        ``channel_wise = True`` (or ``run_once = True``) are merged into a
        single ``'channel_sequential'`` group.  This grouping is entirely
        independent of ``parallel_safe``.
        """
        groups: list[tuple[list[Processor], str]] = []
        i = 0
        while i < len(self.processors):
            proc = self.processors[i]
            ch_eligible = getattr(proc, "channel_wise", False) or getattr(proc, "run_once", False)
            if channel_sequential and ch_eligible:
                batch: list[Processor] = []
                while i < len(self.processors):
                    p = self.processors[i]
                    if getattr(p, "channel_wise", False) or getattr(p, "run_once", False):
                        batch.append(p)
                        i += 1
                    else:
                        break
                groups.append((batch, "channel_sequential"))
            elif parallel and proc.parallel_safe:
                groups.append(([proc], "parallel"))
                i += 1
            else:
                groups.append(([proc], "serial"))
                i += 1
        return groups

    def _dispatch_step(
        self,
        processors: list[Processor],
        mode: str,
        context: ProcessingContext,
        n_jobs: int,
    ) -> ProcessingContext:
        """Execute one group of processors according to *mode*."""
        if mode == "channel_sequential":
            return ChannelSequentialExecutor().execute(processors, context)
        if mode == "parallel":
            return ParallelExecutor(n_jobs=n_jobs).execute(processors[0], context)
        return processors[0].execute(context)

    # ---------------------------------------------------------------------- #
    # Public API                                                              #
    # ---------------------------------------------------------------------- #

    def run(
        self,
        initial_context: ProcessingContext | None = None,
        parallel: bool = False,
        n_jobs: int = -1,
        channel_sequential: bool = False,
        show_progress: bool = True,
    ) -> PipelineResult:
        """
        Execute the pipeline.

        Args:
            initial_context: Initial context (if None, first processor creates it)
            parallel: Enable parallel execution for compatible processors
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            channel_sequential: Run consecutive channel-wise processors
                (``channel_wise = True``) as a single per-channel pass.
                For each channel the full local sequence executes before
                the next channel starts::

                    for each channel:
                        channel → HP-filter → UpSample → AAS → DownSample → store

                The output array is pre-allocated at the final sfreq so the
                full high-sfreq intermediate data never exists for all
                channels at once.

                Processors with ``run_once = True`` (e.g. TriggerAligner)
                are included in the per-channel pass but only execute for
                the first channel; all subsequent channels skip them.

                This flag is independent of ``parallel_safe`` and has no
                relation to multiprocessing.  Takes precedence over
                *parallel* for eligible processors.
            show_progress: Show progress bar

        Returns:
            PipelineResult containing final context and metadata
        """
        import time

        start_time = time.time()
        console = get_console()
        n_procs = len(self.processors)

        execution_mode = "channel_sequential" if channel_sequential else "parallel" if parallel else "serial"
        console.set_pipeline_metadata(
            {
                "execution_mode": execution_mode,
                "n_jobs": "1" if channel_sequential else str(n_jobs),
            }
        )
        console.start_pipeline(
            self.name,
            n_procs,
            step_names=[p.name for p in self.processors],
        )
        logger.info(f"Starting pipeline: {self.name} ({n_procs} processors)")

        context = initial_context
        current_processor: tuple[int, Processor] | None = None

        try:
            step_offset = 0
            for processors, mode in self._group_processors(parallel, channel_sequential):
                current_processor = (step_offset, processors[0])

                label = " → ".join(p.name for p in processors)
                logger.info(f"[{step_offset + 1}/{n_procs}] {label}")
                for k, p in enumerate(processors):
                    console.step_started(step_offset + k, p.name)

                set_current_step_index(step_offset)
                step_start = time.time()
                try:
                    context = self._dispatch_step(processors, mode, context, n_jobs)
                finally:
                    set_current_step_index(None)

                duration = time.time() - step_start
                for k, p in enumerate(processors):
                    console.step_completed(
                        step_offset + k,
                        p.name,
                        duration / len(processors),
                        metrics={
                            "execution_mode": mode,
                            "last_duration": f"{duration:.2f}s",
                        },
                    )
                step_offset += len(processors)

            execution_time = time.time() - start_time
            logger.info(f"Pipeline completed in {execution_time:.2f}s")
            console.pipeline_complete(True, execution_time)
            return PipelineResult(context=context, success=True, execution_time=execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            if current_processor:
                failed_idx, failed_proc = current_processor
                logger.error(
                    f"Pipeline failed after {execution_time:.2f}s during "
                    f"{failed_proc.name} (step {failed_idx + 1}/{n_procs}): {e}"
                )
            else:
                logger.error(f"Pipeline failed after {execution_time:.2f}s: {e}")
            logger.opt(exception=e).debug("Exception details")

            console.pipeline_failed(
                execution_time,
                e,
                current_processor[0] if current_processor else None,
                current_processor[1].name if current_processor else None,
            )
            return PipelineResult(
                context=context,
                success=False,
                error=e,
                execution_time=execution_time,
                failed_processor=current_processor[1].name if current_processor else None,
                failed_processor_index=current_processor[0] if current_processor else None,
            )

    def add(self, processor: Processor | Callable) -> "Pipeline":
        """
        Add a processor or callable to the pipeline (fluent API).

        Args:
            processor: :class:`~facet.core.Processor` instance or callable.

        Returns:
            Self for chaining
        """
        [normalised] = self._normalise_processors([processor], _index_offset=len(self.processors))
        self.processors.append(normalised)
        return self

    def extend(self, processors: list[Processor | Callable]) -> "Pipeline":
        """
        Extend pipeline with multiple processors or callables.

        Args:
            processors: List of processors or callables to add.

        Returns:
            Self for chaining
        """
        self.processors.extend(self._normalise_processors(processors, _index_offset=len(self.processors)))
        return self

    def insert(self, index: int, processor: Processor | Callable) -> "Pipeline":
        """
        Insert a processor or callable at a specific position.

        Args:
            index: Position to insert
            processor: :class:`~facet.core.Processor` instance or callable.

        Returns:
            Self for chaining
        """
        [normalised] = self._normalise_processors([processor], _index_offset=index)
        self.processors.insert(index, normalised)
        return self

    def remove(self, index: int) -> "Pipeline":
        """
        Remove processor at index.

        Args:
            index: Index to remove

        Returns:
            Self for chaining
        """
        self.processors.pop(index)
        return self

    def validate_all(self, context: ProcessingContext) -> list[str]:
        """
        Validate all processors against a context.

        Useful for checking if a pipeline can run before actually running it.

        Args:
            context: Context to validate against

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        for i, processor in enumerate(self.processors):
            try:
                processor.validate(context)
            except Exception as e:
                errors.append(f"Processor {i} ({processor.name}): {str(e)}")
        return errors

    def describe(self) -> str:
        """
        Get human-readable pipeline description.

        Returns:
            Multi-line string describing pipeline
        """
        lines = [f"Pipeline: {self.name}", "=" * 50]

        for i, processor in enumerate(self.processors):
            lines.append(f"{i + 1}. {processor.name} ({processor.__class__.__name__})")
            if hasattr(processor, "_parameters"):
                for key, value in processor._parameters.items():
                    lines.append(f"   - {key}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize pipeline to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "processors": [
                {
                    "class": proc.__class__.__name__,
                    "name": proc.name,
                    "parameters": proc._parameters if hasattr(proc, "_parameters") else {},
                }
                for proc in self.processors
            ],
        }

    def map(
        self,
        inputs: list[str | ProcessingContext],
        loader_factory: Callable[[str], "Processor"] | None = None,
        parallel: bool = False,
        n_jobs: int = -1,
        on_error: str = "continue",
        keep_raw: bool = True,
    ) -> "BatchResult":
        """
        Run the pipeline on multiple inputs and return a result per input.

        Each input can be:

        - A ``ProcessingContext`` — passed directly as ``initial_context``.
        - A **file path string** — a fresh :class:`~facet.io.Loader` is created
          automatically for each path via *loader_factory*.

        .. note::
            Do **not** add a :class:`~facet.io.Loader` processor to the pipeline
            when using ``map()``.  Loading is handled outside the pipeline so
            that each file gets its own isolated loader instance.

        Args:
            inputs: List of file paths or ``ProcessingContext`` objects.
            loader_factory: ``Callable[[path], Processor]`` that creates a fresh
                loader for each path string.  Defaults to
                ``lambda p: Loader(path=p, preload=True)``.
            parallel: Whether to pass ``parallel=True`` to each ``pipeline.run()``.
            n_jobs: Passed through to ``pipeline.run()``.
            on_error: ``"continue"`` (default) — log failures and keep going;
                      ``"raise"`` — re-raise the first error encountered.
            keep_raw: If ``False``, the Raw data is released from each result
                after the pipeline run completes, keeping only metrics and
                history in memory.  Set to ``False`` when processing many files
                and you only need summary statistics.  Defaults to ``True``.

        Returns:
            :class:`BatchResult` containing one :class:`PipelineResult` per
            input, in the same order.  It behaves like a plain list but also
            offers :meth:`~BatchResult.print_summary` and
            :attr:`~BatchResult.summary_df`.

        Example::

            pipeline = Pipeline([
                TriggerDetector(regex=r"\\b1\\b"),
                UpSample(factor=10),
                AASCorrection(window_size=30),
                DownSample(factor=10),
                SNRCalculator(),
            ])

            results = pipeline.map(
                ["sub-01.edf", "sub-02.edf", "sub-03.edf"],
                keep_raw=False,
            )
            results.print_summary()
        """
        from ..io.loaders import Loader as _Loader

        if loader_factory is None:
            loader_factory = lambda p: _Loader(path=p, preload=True)  # noqa: E731

        for proc in self.processors:
            if isinstance(proc, _Loader):
                raise ValueError(
                    "A Loader processor was found inside the pipeline passed to map(). "
                    "map() handles loading automatically — remove the Loader from the "
                    "pipeline and pass file paths directly to map()."
                )

        results: list[PipelineResult] = []
        labels: list[str] = []

        for item in inputs:
            if isinstance(item, ProcessingContext):
                run_pipeline = self
                initial_ctx = item
                label = repr(item)
            else:
                label = str(item)
                initial_ctx = None

                loader = loader_factory(item)
                try:
                    initial_ctx = loader.execute(None)
                except Exception as exc:
                    logger.error(f"Loader failed for '{label}': {exc}")
                    result = PipelineResult(
                        context=None,
                        success=False,
                        error=exc,
                        failed_processor=getattr(loader, "name", "loader"),
                    )
                    results.append(result)
                    labels.append(label)
                    if on_error == "raise":
                        raise
                    continue
                run_pipeline = self

            logger.info(f"Pipeline.map: processing '{label}'")

            result = run_pipeline.run(
                initial_context=initial_ctx,
                parallel=parallel,
                n_jobs=n_jobs,
            )

            if not keep_raw and result.success:
                result.release_raw()

            results.append(result)
            labels.append(label)

            if not result.success and on_error == "raise":
                raise result.error

        return BatchResult(results, labels=labels)

    def __len__(self) -> int:
        """Get number of processors."""
        return len(self.processors)

    def __getitem__(self, index: int) -> Processor:
        """Get processor by index."""
        return self.processors[index]

    def __repr__(self) -> str:
        """String representation."""
        return f"Pipeline(name='{self.name}', n_processors={len(self.processors)})"


class PipelineBuilder:
    """
    Fluent builder for constructing pipelines.

    Example::

        pipeline = (PipelineBuilder()
            .add(Loader("data.edf"))
            .highpass(1.0)
            .upsample(10)
            .detect_triggers(r"\\btrigger\\b")
            .aas_correction()
            .export_edf("output.edf")
            .build())
    """

    def __init__(self, name: str | None = None):
        """
        Initialize builder.

        Args:
            name: Optional pipeline name
        """
        self._processors: list[Processor] = []
        self._name = name

    def add(self, processor: Processor) -> "PipelineBuilder":
        """
        Add custom processor.

        Args:
            processor: Processor to add

        Returns:
            Self for chaining
        """
        self._processors.append(processor)
        return self

    def add_if(self, condition: bool, processor: Processor) -> "PipelineBuilder":
        """
        Add processor conditionally.

        Args:
            condition: Whether to add processor
            processor: Processor to add

        Returns:
            Self for chaining
        """
        if condition:
            self._processors.append(processor)
        return self

    def build(self) -> Pipeline:
        """
        Build the pipeline.

        Returns:
            Constructed Pipeline instance
        """
        return Pipeline(self._processors, name=self._name)

    # Convenience methods can be added here for common processors
    # These will be populated as we implement specific processors
