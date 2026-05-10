"""Training configuration dataclasses with YAML/JSON round-trip serialization."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConfig:
    """Controls how model checkpoints are saved during training.

    Parameters
    ----------
    dirpath : str
        Directory where checkpoint files are written.
    monitor : str
        Metric name to optimise (must match a key returned by eval_step).
    mode : {"min", "max"}
        Whether lower or higher values of *monitor* are better.
    save_top_k : int
        Keep at most this many best checkpoints on disk (-1 = keep all).
    save_last : bool
        Always write a ``last.ckpt`` symlink/copy after each epoch.
    filename_template : str
        f-string template for checkpoint filenames.  Available variables:
        ``epoch``, ``monitor`` (metric name, snake_cased), ``value``.
    """

    dirpath: str = "checkpoints"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 3
    save_last: bool = True
    filename_template: str = "epoch{epoch:04d}_{monitor}{value:.4f}"


@dataclass
class EarlyStoppingConfig:
    """Controls early stopping.

    Parameters
    ----------
    monitor : str
        Metric to watch.
    mode : {"min", "max"}
        Whether lower or higher values are better.
    patience : int
        Number of epochs without improvement before training stops.
    min_delta : float
        Minimum change in *monitor* to count as an improvement.
    """

    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class AugmentationConfig:
    """Data augmentation applied to each training batch.

    All augmentations operate on numpy arrays and are framework-agnostic.

    Parameters
    ----------
    enabled : bool
        Master switch.  When False all augmentations are skipped.
    trigger_jitter_samples : int
        ±samples of random jitter added to each chunk's start position.
    noise_scale_range : tuple[float, float]
        Multiplicative Gaussian noise drawn uniformly from this range.
        ``(1.0, 1.0)`` disables noise scaling.
    channel_dropout_prob : float
        Probability of zeroing an individual channel per batch item.
    sign_flip_prob : float
        Probability of flipping the polarity of a batch item.
    """

    enabled: bool = False
    trigger_jitter_samples: int = 0
    noise_scale_range: tuple[float, float] = (1.0, 1.0)
    channel_dropout_prob: float = 0.0
    sign_flip_prob: float = 0.0


@dataclass
class LoggingConfig:
    """Controls what gets logged and where.

    Parameters
    ----------
    log_every_n_steps : int
        Log training metrics every N batch steps.
    val_every_n_epochs : int
        Run validation every N epochs.
    rich_live : bool
        Show the Rich live dashboard in the terminal.
    log_file : str or None
        Path to a JSONL file for persistent metric logging.  ``None``
        disables file logging.
    loss_plot_file : str or None
        Path to a PNG file for the per-epoch training/validation loss plot.
        Relative paths are resolved inside the training run directory.
        ``None`` disables plot generation.
    progress_bar : bool
        Show per-batch progress bar inside the live dashboard.
    """

    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    rich_live: bool = True
    log_file: str | None = "training.jsonl"
    loss_plot_file: str | None = "loss.png"
    progress_bar: bool = True


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Complete, serialisable training configuration.

    Intended to be written to / read from YAML or JSON so that every
    training run is fully reproducible.

    Parameters
    ----------
    model_name : str
        Descriptive name used in logs and checkpoint filenames.
    chunk_size : int
        Samples per training chunk (e.g. 1250 = 5 s @ 250 Hz).
    val_ratio : float
        Fraction of contexts (or chunks) used for validation.
    trigger_aligned : bool
        Align chunk boundaries to trigger positions (TR onsets).
    overlap : float
        Overlap ratio between successive sliding-window chunks [0, 1).
    target_type : {"clean", "artifact"}
        What the model should predict: the clean signal or the artifact.
    max_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Number of chunks per mini-batch.
    learning_rate : float
        Initial learning rate passed to the framework-specific wrapper.
    weight_decay : float
        L2 regularisation coefficient.
    grad_clip_norm : float or None
        Maximum gradient L2 norm for gradient clipping.  ``None`` disables.
    seed : int
        Global random seed for reproducibility.
    run_name : str or None
        Human-readable run identifier.  Auto-generated from timestamp when
        ``None``.
    output_dir : str
        Root directory for all training outputs (checkpoints, logs).
    checkpoint : CheckpointConfig
    early_stopping : EarlyStoppingConfig or None
        ``None`` disables early stopping.
    augmentation : AugmentationConfig
    logging : LoggingConfig
    extra : dict
        Arbitrary extra fields forwarded to framework-specific wrappers.
    """

    # Model
    model_name: str = "custom"

    # Data
    chunk_size: int = 1250
    val_ratio: float = 0.2
    trigger_aligned: bool = True
    overlap: float = 0.0
    target_type: str = "clean"

    # Optimisation
    max_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 1.0
    seed: int = 42

    # Identity
    run_name: str | None = None
    output_dir: str = "training_output"

    # Sub-configs
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig | None = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Escape hatch for framework-specific params
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (JSON-serialisable)."""
        raw = asdict(self)
        # Convert tuple fields to lists for JSON compatibility
        aug = raw.get("augmentation", {})
        if isinstance(aug.get("noise_scale_range"), tuple):
            aug["noise_scale_range"] = list(aug["noise_scale_range"])
        return raw

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Reconstruct from a plain dict."""
        data = dict(data)

        # Nested dataclass reconstruction
        if "checkpoint" in data and isinstance(data["checkpoint"], dict):
            data["checkpoint"] = CheckpointConfig(**data["checkpoint"])

        if "early_stopping" in data and isinstance(data["early_stopping"], dict):
            data["early_stopping"] = EarlyStoppingConfig(**data["early_stopping"])

        if "augmentation" in data and isinstance(data["augmentation"], dict):
            aug = data["augmentation"]
            if "noise_scale_range" in aug and isinstance(aug["noise_scale_range"], list):
                aug["noise_scale_range"] = tuple(aug["noise_scale_range"])
            data["augmentation"] = AugmentationConfig(**aug)

        if "logging" in data and isinstance(data["logging"], dict):
            data["logging"] = LoggingConfig(**data["logging"])

        # Drop unknown keys for forward compatibility
        known = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in data.items() if k in known}

        return cls(**data)

    def save_yaml(self, path: str | Path) -> None:
        """Write config to a YAML file."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyYAML is required for YAML serialisation: pip install pyyaml") from exc

        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load config from a YAML file."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyYAML is required for YAML serialisation: pip install pyyaml") from exc

        with Path(path).expanduser().open("r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def save_json(self, path: str | Path) -> None:
        """Write config to a JSON file."""
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> TrainingConfig:
        """Load config from a JSON file."""
        with Path(path).expanduser().open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
