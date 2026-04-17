"""Tests for the ``facet-train`` command-line entry point."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from facet.training import cli as training_cli

TEST_MODULE_SOURCE = """
from __future__ import annotations

import mne
import numpy as np

from facet.core import ProcessingContext, ProcessingMetadata


class DummyOptimizer:
    pass


class DummyScheduler:
    pass


def build_contexts(training_config=None):
    sfreq = 250.0
    n_samples = 1000
    times = np.arange(n_samples) / sfreq
    clean = np.vstack([
        np.sin(2 * np.pi * 8.0 * times),
        np.sin(2 * np.pi * 12.0 * times + 0.3),
    ]).astype(np.float32)
    artifact = np.zeros_like(clean)
    trigger_starts = np.array([0, 250, 500, 750], dtype=int)
    for start in trigger_starts:
        artifact[:, start:start + 40] += 0.2

    info = mne.create_info(
        ch_names=["EEG001", "EEG002"],
        sfreq=sfreq,
        ch_types=["eeg", "eeg"],
    )
    metadata = ProcessingMetadata()
    metadata.triggers = trigger_starts
    metadata.artifact_length = 40
    return [
        ProcessingContext(
            raw=mne.io.RawArray(clean + artifact, info, verbose=False),
            raw_original=mne.io.RawArray(clean, info, verbose=False),
            metadata=metadata,
        )
    ]


def build_model(n_channels, chunk_size, scale=1.0, training_config=None):
    return {
        "n_channels": n_channels,
        "chunk_size": chunk_size,
        "scale": scale,
        "run_name": training_config.run_name if training_config else None,
    }


def build_loss(scale=1.0):
    def _loss(prediction, target):
        return scale
    return _loss
"""


class FakePyTorchWrapper:
    """Wrapper stub used to test the CLI without a torch dependency."""

    last_instance: FakePyTorchWrapper | None = None

    def __init__(
        self,
        *,
        model,
        loss_fn,
        device="cpu",
        optimizer_cls=None,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.train_calls = 0
        self.eval_calls = 0
        FakePyTorchWrapper.last_instance = self

    def train_step(self, noisy, target):
        self.train_calls += 1
        return {"loss": float(((noisy - target) ** 2).mean())}

    def eval_step(self, noisy, target):
        self.eval_calls += 1
        return {"loss": float(((noisy - target) ** 2).mean())}

    def save_checkpoint(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-checkpoint", encoding="utf-8")

    def load_checkpoint(self, path: Path):
        return None

    @property
    def device_info(self) -> str:
        return self.device


def _write_test_module(tmp_path: Path, module_name: str = "cli_training_module") -> str:
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(TEST_MODULE_SOURCE, encoding="utf-8")
    return module_name


def _write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "train_config.json"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


@pytest.mark.unit
def test_run_fit_command_executes_training_and_writes_summary(tmp_path, monkeypatch):
    module_name = _write_test_module(tmp_path)
    monkeypatch.setattr(training_cli, "PyTorchModelWrapper", FakePyTorchWrapper)

    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "kwargs": {"scale": 2.0},
                "loss_factory": f"{module_name}:build_loss",
                "loss_kwargs": {"scale": 0.5},
                "optimizer_factory": f"{module_name}:DummyOptimizer",
                "scheduler_factory": f"{module_name}:DummyScheduler",
                "device": "cpu",
            },
            "data": {
                "context_factory": f"{module_name}:build_contexts",
            },
            "training": {
                "model_name": "CLITestModel",
                "chunk_size": 250,
                "target_type": "artifact",
                "trigger_aligned": True,
                "val_ratio": 0.25,
                "max_epochs": 2,
                "batch_size": 2,
                "output_dir": str(tmp_path / "runs"),
            },
            "checkpoint": {
                "monitor": "loss",
                "save_top_k": 1,
                "save_last": True,
            },
            "logging": {
                "rich_live": False,
                "log_file": "metrics.jsonl",
            },
            "export": {
                "enabled": False,
            },
        },
    )

    run = training_cli.run_fit_command(config_path)

    assert run.result.success is True
    assert run.export_path is None
    assert run.summary_path is not None
    assert run.summary_path.exists()
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert summary["framework"] == "pytorch"
    assert summary["dataset"]["n_channels"] == 2
    assert summary["dataset"]["n_train_chunks"] > 0
    assert (run.result.run_dir / "facet_train_config.resolved.json").exists()
    if importlib.util.find_spec("yaml") is not None:
        assert (run.result.run_dir / "facet_train_config.resolved.yaml").exists()

    wrapper = FakePyTorchWrapper.last_instance
    assert wrapper is not None
    assert wrapper.optimizer_cls.__name__ == "DummyOptimizer"
    assert wrapper.scheduler_cls.__name__ == "DummyScheduler"
    assert wrapper.train_calls > 0
    assert wrapper.eval_calls > 0


@pytest.mark.unit
def test_run_fit_command_writes_inference_config_when_export_enabled(tmp_path, monkeypatch):
    module_name = _write_test_module(tmp_path, module_name="cli_training_module_export")
    monkeypatch.setattr(training_cli, "PyTorchModelWrapper", FakePyTorchWrapper)

    def _fake_export(cli_config, wrapper, dataset, run_dir):
        export_path = run_dir / "exports" / "model.ts"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text("fake-torchscript", encoding="utf-8")
        return export_path

    monkeypatch.setattr(training_cli, "_export_model_if_requested", _fake_export)

    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "loss_factory": f"{module_name}:build_loss",
            },
            "data": {
                "context_factory": f"{module_name}:build_contexts",
            },
            "training": {
                "model_name": "CLIInferenceModel",
                "chunk_size": 250,
                "target_type": "artifact",
                "trigger_aligned": True,
                "val_ratio": 0.25,
                "max_epochs": 1,
                "batch_size": 2,
                "output_dir": str(tmp_path / "runs"),
            },
            "checkpoint": {
                "monitor": "loss",
                "save_top_k": 1,
                "save_last": False,
            },
            "logging": {
                "rich_live": False,
                "log_file": None,
            },
            "export": {
                "enabled": True,
                "write_inference_config": True,
            },
            "inference": {
                "name": "CLIInferenceModel",
                "architecture": "autoencoder",
                "output_type": "artifact",
                "execution_granularity": "channel",
                "supports_multichannel": False,
                "device_preference": "cpu",
            },
        },
    )

    run = training_cli.run_fit_command(config_path)

    assert run.export_path is not None
    assert run.export_path.exists()
    assert run.inference_config_path is not None
    assert run.inference_config_path.exists()

    saved = json.loads(run.inference_config_path.read_text(encoding="utf-8"))
    assert saved["adapter"] == "pytorch_inference"
    assert saved["spec"]["checkpoint_path"] == str(run.export_path)


@pytest.mark.unit
def test_run_fit_command_defaults_inference_output_type_from_clean_target(tmp_path, monkeypatch):
    module_name = _write_test_module(tmp_path, module_name="cli_training_module_clean")
    monkeypatch.setattr(training_cli, "PyTorchModelWrapper", FakePyTorchWrapper)

    def _fake_export(cli_config, wrapper, dataset, run_dir):
        export_path = run_dir / "exports" / "model.ts"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text("fake-torchscript", encoding="utf-8")
        return export_path

    monkeypatch.setattr(training_cli, "_export_model_if_requested", _fake_export)

    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "loss_factory": f"{module_name}:build_loss",
            },
            "data": {
                "context_factory": f"{module_name}:build_contexts",
            },
            "training": {
                "model_name": "CLICleanInferenceModel",
                "chunk_size": 250,
                "target_type": "clean",
                "trigger_aligned": True,
                "val_ratio": 0.25,
                "max_epochs": 1,
                "batch_size": 2,
                "output_dir": str(tmp_path / "runs"),
            },
            "checkpoint": {
                "monitor": "loss",
                "save_top_k": 1,
                "save_last": False,
            },
            "logging": {
                "rich_live": False,
                "log_file": None,
            },
            "export": {
                "enabled": True,
                "write_inference_config": True,
            },
            "inference": {
                "name": "CLICleanInferenceModel",
                "architecture": "autoencoder",
                "execution_granularity": "channel",
                "supports_multichannel": False,
                "device_preference": "cpu",
            },
        },
    )

    run = training_cli.run_fit_command(config_path)
    saved = json.loads(run.inference_config_path.read_text(encoding="utf-8"))
    assert saved["spec"]["output_type"] == "clean"


@pytest.mark.unit
def test_run_fit_command_rejects_inconsistent_inference_output_type(tmp_path, monkeypatch):
    module_name = _write_test_module(tmp_path, module_name="cli_training_module_invalid")
    monkeypatch.setattr(training_cli, "PyTorchModelWrapper", FakePyTorchWrapper)

    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "loss_factory": f"{module_name}:build_loss",
            },
            "data": {
                "context_factory": f"{module_name}:build_contexts",
            },
            "training": {
                "model_name": "CLIInvalidInferenceModel",
                "chunk_size": 250,
                "target_type": "artifact",
                "trigger_aligned": True,
                "val_ratio": 0.25,
                "max_epochs": 1,
                "batch_size": 2,
                "output_dir": str(tmp_path / "runs"),
            },
            "checkpoint": {
                "monitor": "loss",
                "save_top_k": 1,
                "save_last": False,
            },
            "logging": {
                "rich_live": False,
                "log_file": None,
            },
            "export": {
                "enabled": False,
            },
            "inference": {
                "name": "CLIInvalidInferenceModel",
                "architecture": "autoencoder",
                "output_type": "clean",
                "execution_granularity": "channel",
                "supports_multichannel": False,
                "device_preference": "cpu",
            },
        },
    )

    with pytest.raises(
        training_cli.ProcessorValidationError,
        match="Inference output_type is inconsistent with training.target_type",
    ):
        training_cli.run_fit_command(config_path)


@pytest.mark.unit
def test_cli_main_fit_returns_zero(tmp_path, monkeypatch):
    module_name = _write_test_module(tmp_path, module_name="cli_training_module_main")
    monkeypatch.setattr(training_cli, "PyTorchModelWrapper", FakePyTorchWrapper)

    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "loss_factory": f"{module_name}:build_loss",
            },
            "data": {
                "context_factory": f"{module_name}:build_contexts",
            },
            "training": {
                "model_name": "CLIMainModel",
                "chunk_size": 250,
                "target_type": "artifact",
                "trigger_aligned": True,
                "val_ratio": 0.0,
                "max_epochs": 1,
                "batch_size": 2,
                "output_dir": str(tmp_path / "runs"),
            },
            "checkpoint": {
                "monitor": "loss",
                "save_top_k": 1,
                "save_last": False,
            },
            "logging": {
                "rich_live": False,
                "log_file": None,
            },
            "export": {
                "enabled": False,
            },
        },
    )

    assert training_cli.main(["fit", "--config", str(config_path)]) == 0
