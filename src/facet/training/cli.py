"""Command-line interface for training deep-learning models with FACETpy."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from ..core import ProcessingContext, ProcessorValidationError
from ..correction import DeepLearningCorrection, save_deep_learning_config
from ..correction.deep_learning import spec_to_dict
from .config import TrainingConfig
from .dataset import ChannelDropout, EEGArtifactDataset, NoiseScaling, SignFlip, TriggerJitter
from .trainer import Trainer, TrainingResult
from .wrapper import PyTorchModelWrapper, TensorFlowModelWrapper


@dataclass
class ModelCLIConfig:
    """Model and wrapper configuration for the training CLI."""

    framework: str
    factory: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    loss_factory: str | None = None
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    optimizer_factory: str | None = None
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    scheduler_factory: str | None = None
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataCLIConfig:
    """Data source configuration for the training CLI."""

    context_factory: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    eeg_only: bool = True


@dataclass
class ExportCLIConfig:
    """Export configuration for the training CLI."""

    enabled: bool = True
    format: str | None = None
    path: str | None = None
    example_input_shape: list[int] | None = None
    write_inference_config: bool = True
    inference_config_path: str | None = None


@dataclass
class TrainingCLIConfig:
    """Resolved CLI configuration for a training run."""

    model: ModelCLIConfig
    data: DataCLIConfig
    training: TrainingConfig
    export: ExportCLIConfig = field(default_factory=ExportCLIConfig)
    inference: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingCLIConfig:
        """Build a CLI config from a nested dict."""
        if "model" not in data:
            raise ProcessorValidationError("Training CLI config requires a 'model' section")
        if "data" not in data:
            raise ProcessorValidationError("Training CLI config requires a 'data' section")

        model = ModelCLIConfig(**data["model"])
        dataset = DataCLIConfig(**data["data"])

        training_dict = dict(data.get("training", {}))
        for key in ("checkpoint", "early_stopping", "augmentation", "logging"):
            if key in data and key not in training_dict:
                training_dict[key] = data[key]
        training = TrainingConfig.from_dict(training_dict)

        export = ExportCLIConfig(**data.get("export", {}))
        inference = data.get("inference")
        return cls(
            model=model,
            data=dataset,
            training=training,
            export=export,
            inference=dict(inference) if isinstance(inference, dict) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible representation."""
        return {
            "model": dataclasses.asdict(self.model),
            "data": dataclasses.asdict(self.data),
            "training": self.training.to_dict(),
            "export": dataclasses.asdict(self.export),
            "inference": dict(self.inference) if self.inference is not None else None,
        }


@dataclass
class CLITrainingRun:
    """Result payload returned by :func:`run_fit_command`."""

    config: TrainingCLIConfig
    result: TrainingResult
    export_path: Path | None = None
    inference_config_path: Path | None = None
    summary_path: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``facet-train`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="facet-train",
        description="Train deep-learning models with FACETpy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Train a model from a YAML/JSON config.")
    fit_parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML or JSON training configuration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``facet-train``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fit":
        run = run_fit_command(args.config)
        print(f"Run directory: {Path(run.result.run_dir).resolve()}")
        print(f"Best epoch: {run.result.best_epoch}")
        print(f"Best metric: {run.result.best_metric:.6f}")
        if run.export_path is not None:
            print(f"Export: {run.export_path.resolve()}")
        if run.inference_config_path is not None:
            print(f"Inference config: {run.inference_config_path.resolve()}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def run_fit_command(config_path: str | Path) -> CLITrainingRun:
    """Run a complete training job from a YAML/JSON config file."""
    config_path = Path(config_path).expanduser().resolve()
    config_dir = config_path.parent

    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))

    cli_config = load_training_cli_config(config_path)
    _validate_training_config(cli_config)

    contexts = _load_contexts(cli_config)
    dataset = _build_dataset(contexts, cli_config)
    if len(dataset) == 0:
        raise ProcessorValidationError("Dataset construction produced zero chunks; training cannot proceed")

    if cli_config.training.val_ratio > 0.0:
        train_dataset, val_dataset = dataset.train_val_split(
            val_ratio=cli_config.training.val_ratio,
            seed=cli_config.training.seed,
        )
    else:
        train_dataset, val_dataset = dataset, None

    sfreq = contexts[0].get_sfreq()
    model = _build_model(cli_config, dataset, sfreq)
    wrapper = _build_wrapper(cli_config, model)

    trainer = Trainer(
        wrapper=wrapper,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cli_config.training,
    )
    result = trainer.fit()

    run_dir = Path(result.run_dir)
    _write_resolved_cli_config(cli_config, run_dir)

    export_path = _export_model_if_requested(
        cli_config=cli_config,
        wrapper=wrapper,
        dataset=dataset,
        run_dir=run_dir,
    )
    inference_config_path = _write_inference_config_if_requested(
        cli_config=cli_config,
        export_path=export_path,
        run_dir=run_dir,
    )
    summary_path = _write_run_summary(
        cli_config=cli_config,
        result=result,
        n_contexts=len(contexts),
        train_chunks=len(train_dataset),
        val_chunks=len(val_dataset) if val_dataset is not None else 0,
        dataset=dataset,
        export_path=export_path,
        inference_config_path=inference_config_path,
    )

    logger.info(
        "Training completed: run_dir={} best_epoch={} best_metric={:.6f}",
        run_dir,
        result.best_epoch,
        result.best_metric,
    )
    return CLITrainingRun(
        config=cli_config,
        result=result,
        export_path=export_path,
        inference_config_path=inference_config_path,
        summary_path=summary_path,
    )


def load_training_cli_config(path: str | Path) -> TrainingCLIConfig:
    """Load a CLI config from YAML or JSON."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required for YAML training configs. Install with: pip install pyyaml"
            ) from exc
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    else:
        raise ProcessorValidationError(
            f"Unsupported config format '{suffix}'. Use .json, .yaml, or .yml"
        )

    if not isinstance(raw, dict):
        raise ProcessorValidationError("Training CLI config must decode to a top-level mapping")
    return TrainingCLIConfig.from_dict(raw)


def _validate_training_config(cli_config: TrainingCLIConfig) -> None:
    framework = cli_config.model.framework.strip().lower()
    if framework not in {"pytorch", "tensorflow"}:
        raise ProcessorValidationError(
            f"Unsupported training framework '{cli_config.model.framework}'. Use 'pytorch' or 'tensorflow'."
        )
    if cli_config.training.val_ratio < 0 or cli_config.training.val_ratio >= 1:
        raise ProcessorValidationError("training.val_ratio must be in the range [0, 1)")
    if not cli_config.model.factory.strip():
        raise ProcessorValidationError("model.factory must be a non-empty 'module:function' reference")
    if not cli_config.data.context_factory.strip():
        raise ProcessorValidationError("data.context_factory must be a non-empty 'module:function' reference")
    if framework == "pytorch" and cli_config.model.loss_factory is None:
        logger.info("No model.loss_factory configured; defaulting to torch.nn.MSELoss")
    _resolve_inference_spec(cli_config)


def _resolve_expected_output_type(target_type: str) -> str:
    mapping = {
        "artifact": "artifact",
        "clean": "clean",
    }
    try:
        return mapping[target_type.strip().lower()]
    except KeyError as exc:
        raise ProcessorValidationError(
            f"Unsupported training target_type '{target_type}'. Use 'artifact' or 'clean'."
        ) from exc


def _resolve_inference_spec(cli_config: TrainingCLIConfig) -> dict[str, Any] | None:
    if cli_config.inference is None:
        return None

    spec = dict(cli_config.inference)
    expected_output_type = _resolve_expected_output_type(cli_config.training.target_type)
    configured_output_type = spec.get("output_type")

    if configured_output_type is None:
        spec["output_type"] = expected_output_type
    elif str(configured_output_type).strip().lower() != expected_output_type:
        raise ProcessorValidationError(
            "Inference output_type is inconsistent with training.target_type: "
            f"training.target_type='{cli_config.training.target_type}' implies "
            f"inference.output_type='{expected_output_type}', got '{configured_output_type}'."
        )

    return spec


def _import_object(spec: str) -> Any:
    if ":" not in spec:
        raise ProcessorValidationError(
            f"Factory reference '{spec}' is invalid. Use the form 'module:function'."
        )
    module_name, attr_path = spec.split(":", maxsplit=1)
    if not module_name or not attr_path:
        raise ProcessorValidationError(
            f"Factory reference '{spec}' is invalid. Use the form 'module:function'."
        )
    module = importlib.import_module(module_name)
    obj = module
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def _invoke_factory(factory: Any, explicit_kwargs: dict[str, Any], injected_kwargs: dict[str, Any]) -> Any:
    explicit_kwargs = dict(explicit_kwargs)
    signature = inspect.signature(factory)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    call_kwargs = dict(explicit_kwargs)
    for key, value in injected_kwargs.items():
        if key in call_kwargs:
            continue
        if key in signature.parameters or accepts_var_kwargs:
            call_kwargs[key] = value

    return factory(**call_kwargs)


def _load_contexts(cli_config: TrainingCLIConfig) -> list[ProcessingContext]:
    factory = _import_object(cli_config.data.context_factory)
    factory_result = _invoke_factory(
        factory,
        cli_config.data.kwargs,
        {"training_config": cli_config.training},
    )

    contexts = [factory_result] if isinstance(factory_result, ProcessingContext) else list(factory_result)

    if not contexts:
        raise ProcessorValidationError("data.context_factory returned no ProcessingContext objects")
    if not all(isinstance(ctx, ProcessingContext) for ctx in contexts):
        raise ProcessorValidationError(
            "data.context_factory must return a ProcessingContext or an iterable of ProcessingContext objects"
        )
    return contexts


def _build_dataset(
    contexts: list[ProcessingContext],
    cli_config: TrainingCLIConfig,
) -> EEGArtifactDataset:
    transforms: list[Any] = []
    augmentation = cli_config.training.augmentation
    if augmentation.enabled:
        if augmentation.trigger_jitter_samples > 0:
            transforms.append(
                TriggerJitter(
                    max_jitter=augmentation.trigger_jitter_samples,
                    seed=cli_config.training.seed,
                )
            )
        if augmentation.noise_scale_range != (1.0, 1.0):
            transforms.append(
                NoiseScaling(
                    scale_range=augmentation.noise_scale_range,
                    seed=cli_config.training.seed,
                )
            )
        if augmentation.channel_dropout_prob > 0.0:
            transforms.append(
                ChannelDropout(
                    p=augmentation.channel_dropout_prob,
                    seed=cli_config.training.seed,
                )
            )
        if augmentation.sign_flip_prob > 0.0:
            transforms.append(
                SignFlip(
                    p=augmentation.sign_flip_prob,
                    seed=cli_config.training.seed,
                )
            )

    return EEGArtifactDataset(
        contexts=contexts,
        chunk_size=cli_config.training.chunk_size,
        target_type=cli_config.training.target_type,
        trigger_aligned=cli_config.training.trigger_aligned,
        overlap=cli_config.training.overlap,
        transforms=transforms,
        eeg_only=cli_config.data.eeg_only,
    )


def _build_model(
    cli_config: TrainingCLIConfig,
    dataset: EEGArtifactDataset,
    sfreq: float,
) -> Any:
    factory = _import_object(cli_config.model.factory)
    injected_kwargs = {
        "n_channels": dataset.n_channels,
        "chunk_size": cli_config.training.chunk_size,
        "sfreq": sfreq,
        "target_type": cli_config.training.target_type,
        "training_config": cli_config.training,
    }
    return _invoke_factory(factory, cli_config.model.kwargs, injected_kwargs)


def _build_wrapper(cli_config: TrainingCLIConfig, model: Any) -> Any:
    framework = cli_config.model.framework.strip().lower()
    optimizer_cls = (
        _import_object(cli_config.model.optimizer_factory)
        if cli_config.model.optimizer_factory
        else None
    )
    scheduler_cls = (
        _import_object(cli_config.model.scheduler_factory)
        if cli_config.model.scheduler_factory
        else None
    )

    shared_kwargs = {
        "learning_rate": cli_config.training.learning_rate,
        "weight_decay": cli_config.training.weight_decay,
        "grad_clip_norm": cli_config.training.grad_clip_norm,
    }

    if framework == "pytorch":
        loss_factory = (
            _import_object(cli_config.model.loss_factory)
            if cli_config.model.loss_factory
            else _import_object("torch.nn:MSELoss")
        )
        loss_fn = _invoke_factory(loss_factory, cli_config.model.loss_kwargs, {})
        return PyTorchModelWrapper(
            model=model,
            loss_fn=loss_fn,
            device=cli_config.model.device or "cpu",
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=cli_config.model.optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=cli_config.model.scheduler_kwargs,
            **shared_kwargs,
        )

    if framework == "tensorflow":
        if cli_config.model.loss_factory is not None:
            logger.warning(
                "model.loss_factory is ignored for tensorflow because the Keras model is expected to be pre-compiled"
            )
        return TensorFlowModelWrapper(
            model=model,
            device=cli_config.model.device,
            **shared_kwargs,
        )

    raise ProcessorValidationError(f"Unsupported framework '{cli_config.model.framework}'")


def _write_resolved_cli_config(cli_config: TrainingCLIConfig, run_dir: Path) -> None:
    resolved = cli_config.to_dict()

    json_path = run_dir / "facet_train_config.resolved.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(resolved, fh, indent=2)

    try:
        import yaml
    except ImportError:  # pragma: no cover
        return

    yaml_path = run_dir / "facet_train_config.resolved.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)


def _resolve_export_path(cli_config: TrainingCLIConfig, run_dir: Path) -> Path:
    export_format = _resolved_export_format(cli_config)
    if cli_config.export.path:
        configured = Path(cli_config.export.path).expanduser()
        return configured if configured.is_absolute() else run_dir / configured

    extension_map = {
        "torchscript": ".ts",
        "keras": ".keras",
    }
    return run_dir / "exports" / f"model{extension_map[export_format]}"


def _resolved_export_format(cli_config: TrainingCLIConfig) -> str:
    if cli_config.export.format is not None:
        return cli_config.export.format.strip().lower()

    framework = cli_config.model.framework.strip().lower()
    return "torchscript" if framework == "pytorch" else "keras"


def _export_model_if_requested(
    cli_config: TrainingCLIConfig,
    wrapper: Any,
    dataset: EEGArtifactDataset,
    run_dir: Path,
) -> Path | None:
    if not cli_config.export.enabled:
        return None

    export_path = _resolve_export_path(cli_config, run_dir)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_format = _resolved_export_format(cli_config)
    framework = cli_config.model.framework.strip().lower()

    if framework == "pytorch":
        if export_format != "torchscript":
            raise ProcessorValidationError(
                f"Unsupported export format '{export_format}' for PyTorch. Use 'torchscript'."
            )
        return _export_pytorch_torchscript(
            wrapper=wrapper,
            path=export_path,
            example_input_shape=cli_config.export.example_input_shape,
            n_channels=dataset.n_channels,
            chunk_size=cli_config.training.chunk_size,
        )

    if framework == "tensorflow":
        if export_format != "keras":
            raise ProcessorValidationError(
                f"Unsupported export format '{export_format}' for TensorFlow. Use 'keras'."
            )
        return _export_tensorflow_keras(wrapper=wrapper, path=export_path)

    raise ProcessorValidationError(f"Unsupported framework '{cli_config.model.framework}'")


def _export_pytorch_torchscript(
    wrapper: Any,
    path: Path,
    example_input_shape: list[int] | None,
    n_channels: int,
    chunk_size: int,
) -> Path:
    model = getattr(wrapper, "model", None)
    if model is None:
        raise ProcessorValidationError("PyTorch export requires wrapper.model to be available")

    import torch

    shape = example_input_shape or [1, n_channels, chunk_size]
    if len(shape) != 3:
        raise ProcessorValidationError("export.example_input_shape must have exactly 3 entries: [batch, channels, time]")

    try:
        first_param = next(model.parameters())
        device = first_param.device
    except StopIteration:
        device = torch.device("cpu")

    example_input = torch.randn(*shape, device=device)
    scripted = torch.jit.trace(model.eval(), example_input)
    scripted.save(str(path))
    return path


def _export_tensorflow_keras(wrapper: Any, path: Path) -> Path:
    model = getattr(wrapper, "model", None)
    if model is None:
        raise ProcessorValidationError("TensorFlow export requires wrapper.model to be available")
    model.save(str(path))
    return path


def _write_inference_config_if_requested(
    cli_config: TrainingCLIConfig,
    export_path: Path | None,
    run_dir: Path,
) -> Path | None:
    if not cli_config.export.write_inference_config or cli_config.inference is None:
        return None
    if export_path is None:
        raise ProcessorValidationError(
            "Cannot write inference config without an exported model artifact. Enable export first."
        )

    framework = cli_config.model.framework.strip().lower()
    adapter_name = {
        "pytorch": "pytorch_inference",
        "tensorflow": "tensorflow_inference",
    }[framework]

    spec = _resolve_inference_spec(cli_config)
    if spec is None:
        return None
    spec.setdefault("name", cli_config.training.model_name)
    spec.setdefault("runtime", framework)
    spec["checkpoint_path"] = str(export_path)

    processor = DeepLearningCorrection.from_config_dict(
        {
            "version": "1",
            "processor": "deep_learning_correction",
            "adapter": adapter_name,
            "spec": spec,
            "store_run_metadata": True,
            "trigger_aligned_chunking": False,
            "triggers_per_chunk": 1,
        }
    )

    if cli_config.export.inference_config_path:
        path = Path(cli_config.export.inference_config_path).expanduser()
        path = path if path.is_absolute() else run_dir / path
    else:
        path = run_dir / "inference_processor.json"
    save_deep_learning_config(processor, path)
    return path


def _write_run_summary(
    cli_config: TrainingCLIConfig,
    result: TrainingResult,
    n_contexts: int,
    train_chunks: int,
    val_chunks: int,
    dataset: EEGArtifactDataset,
    export_path: Path | None,
    inference_config_path: Path | None,
) -> Path:
    run_dir = Path(result.run_dir)
    summary = {
        "framework": cli_config.model.framework,
        "model_factory": cli_config.model.factory,
        "context_factory": cli_config.data.context_factory,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "n_contexts": n_contexts,
        "dataset": {
            "n_channels": dataset.n_channels,
            "chunk_size": dataset.chunk_size,
            "n_total_chunks": dataset.n_chunks,
            "n_train_chunks": train_chunks,
            "n_val_chunks": val_chunks,
            "target_type": dataset.target_type,
            "trigger_aligned": dataset.trigger_aligned,
        },
        "training": {
            "total_epochs": result.total_epochs,
            "best_epoch": result.best_epoch,
            "best_metric": result.best_metric,
            "elapsed_seconds": result.elapsed_seconds,
        },
        "export": {
            "enabled": cli_config.export.enabled,
            "path": str(export_path) if export_path is not None else None,
            "format": _resolved_export_format(cli_config) if cli_config.export.enabled else None,
            "inference_config_path": str(inference_config_path) if inference_config_path is not None else None,
        },
        "inference_spec": (
            _summarize_inference_spec(
                framework=cli_config.model.framework,
                inference_spec=_resolve_inference_spec(cli_config),
                export_path=export_path,
            )
            if _resolve_inference_spec(cli_config) is not None and export_path is not None
            else None
        ),
    }

    path = run_dir / "summary.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return path


def _summarize_inference_spec(
    framework: str,
    inference_spec: dict[str, Any],
    export_path: Path,
) -> dict[str, Any]:
    spec = dict(inference_spec)
    spec.setdefault("checkpoint_path", str(export_path))
    spec.setdefault("runtime", framework)
    processor = DeepLearningCorrection.from_config_dict(
        {
            "version": "1",
            "processor": "deep_learning_correction",
            "adapter": (
                "pytorch_inference"
                if str(framework).lower() == "pytorch"
                else "tensorflow_inference"
            ),
            "spec": spec,
            "store_run_metadata": True,
            "trigger_aligned_chunking": False,
            "triggers_per_chunk": 1,
        }
    )
    return spec_to_dict(processor.model.spec)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
