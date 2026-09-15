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


class ContextDataset:
    def __init__(self):
        self.noisy = np.zeros((6, 7, 1, 12), dtype=np.float32)
        self.target = np.ones((6, 1, 12), dtype=np.float32) * 0.25
        self.n_channels = 1
        self.chunk_size = 12
        self.n_chunks = 6
        self.context_epochs = 7
        self.epoch_samples = 12
        self.input_shape = (7, 1, 12)
        self.target_shape = (1, 12)
        self.target_type = "artifact"
        self.trigger_aligned = True
        self.sfreq = 2048.0

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.target[idx]

    def train_val_split(self, val_ratio=0.2, seed=42):
        return self, self


def build_context_dataset(training_config=None):
    return ContextDataset()
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
        prediction = noisy[:, noisy.shape[1] // 2] if noisy.ndim == 4 else noisy
        return {"loss": float(((prediction - target) ** 2).mean())}

    def eval_step(self, noisy, target):
        self.eval_calls += 1
        prediction = noisy[:, noisy.shape[1] // 2] if noisy.ndim == 4 else noisy
        return {"loss": float(((prediction - target) ** 2).mean())}

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
def test_run_fit_command_accepts_dataset_factory_for_context_inputs(tmp_path, monkeypatch):
    module_name = _write_test_module(tmp_path, module_name="cli_training_context_dataset")
    monkeypatch.setattr(training_cli, "PyTorchModelWrapper", FakePyTorchWrapper)

    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "loss_factory": f"{module_name}:build_loss",
                "device": "cpu",
            },
            "data": {
                "dataset_factory": f"{module_name}:build_context_dataset",
            },
            "training": {
                "model_name": "CLIContextDatasetModel",
                "chunk_size": 12,
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
        },
    )

    run = training_cli.run_fit_command(config_path)

    assert run.result.success is True
    assert run.summary_path is not None
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert summary["n_contexts"] == 0
    assert summary["dataset_factory"] == f"{module_name}:build_context_dataset"
    assert summary["dataset"]["input_shape"] == [7, 1, 12]
    assert summary["dataset"]["target_shape"] == [1, 12]

    wrapper = FakePyTorchWrapper.last_instance
    assert wrapper is not None
    assert wrapper.train_calls > 0


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


# ---------------------------------------------------------------------------
# Custom wrapper factory (adversarial training escape hatch)
# ---------------------------------------------------------------------------

ADVERSARIAL_MODULE_SOURCE = '''
from __future__ import annotations

import numpy as np
import torch

from facet.training.adversarial import (
    AdversarialModelWrapper,
    AdversarialObjective,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
)


class Generator(torch.nn.Module):
    """Two named heads over a (batch, epochs, channels, samples) context."""

    def __init__(self, context_epochs=3, n_channels=2, chunk_size=12):
        super().__init__()
        self.context_epochs = context_epochs
        self.n_channels = n_channels
        self.body = torch.nn.Conv1d(context_epochs * n_channels, n_channels, 3, padding=1)
        self.noise_head = torch.nn.Conv1d(n_channels, n_channels, 1)

    def forward(self, x):
        b, ep, ch, t = x.shape
        h = self.body(x.reshape(b, ep * ch, t))
        return {"clean": h, "noise": self.noise_head(h)}


class Discriminator(torch.nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.conv = torch.nn.Conv1d(channels, 4, 3, padding=1)
        self.out = torch.nn.Conv1d(4, 1, 1)

    def forward(self, x):
        h = torch.nn.functional.leaky_relu(self.conv(x), 0.2)
        return self.out(h), [h]


class Objective(AdversarialObjective):
    def __init__(self, channels=2):
        super().__init__()
        self.register_discriminator("clean", Discriminator(channels))
        self.register_discriminator("noise", Discriminator(channels))

    def discriminator_losses(self, outputs, target):
        out = {}
        for name, disc in self.discriminators.items():
            d_real, _ = disc(target)
            d_fake, _ = disc(outputs[name])
            out[name] = lsgan_discriminator_loss(d_real, d_fake)
        return out

    def generator_loss(self, outputs, target):
        total = torch.zeros((), device=target.device)
        for name, disc in self.discriminators.items():
            d_fake, _ = disc(outputs[name])
            total = total + torch.nn.functional.mse_loss(outputs[name], target)
            total = total + 0.1 * lsgan_generator_loss(d_fake)
        return total, {}

    def primary_output(self, outputs):
        return outputs["clean"]


def build_model(context_epochs=3, n_channels=2, chunk_size=12, **_):
    return Generator(context_epochs=context_epochs, n_channels=n_channels, chunk_size=chunk_size)


def build_wrapper(model, device="cpu", learning_rate=1e-3, weight_decay=0.0, grad_clip_norm=1.0, **_):
    return AdversarialModelWrapper(
        model=model,
        objective=Objective(channels=model.n_channels),
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        discriminator_learning_rate=1e-4,
    )


class ContextDataset:
    """(examples, epochs, channels, samples) in, (channels, samples) out."""

    context_epochs = 3
    n_channels = 2
    epoch_samples = 12
    input_shape = (3, 2, 12)
    target_shape = (2, 12)
    sfreq = 250.0

    def __init__(self, n=8):
        rng = np.random.default_rng(0)
        self.noisy = rng.standard_normal((n, 3, 2, 12)).astype(np.float32)
        self.target = rng.standard_normal((n, 2, 12)).astype(np.float32)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.noisy[index], self.target[index]

    def train_val_split(self, val_ratio=0.2, seed=42):
        return self, self


def build_dataset(**_):
    return ContextDataset()
'''


@pytest.mark.unit
def test_run_fit_command_uses_a_custom_wrapper_factory(tmp_path):
    """The escape hatch for models the one-optimizer contract cannot express.

    Trains a two-head generator with two discriminators end-to-end through the CLI
    and asserts that the discriminators really were trained and checkpointed —
    which is exactly what the loss-module GAN workaround could not deliver.
    """
    pytest.importorskip("torch")
    import torch

    module_name = "cli_training_adversarial"
    (tmp_path / f"{module_name}.py").write_text(ADVERSARIAL_MODULE_SOURCE, encoding="utf-8")

    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "wrapper_factory": f"{module_name}:build_wrapper",
                "device": "cpu",
            },
            "data": {"dataset_factory": f"{module_name}:build_dataset"},
            "training": {
                "model_name": "CLIAdversarialModel",
                "chunk_size": 12,
                "target_type": "clean",
                "val_ratio": 0.25,
                "max_epochs": 1,
                "batch_size": 4,
                "output_dir": str(tmp_path / "runs"),
            },
            "checkpoint": {"monitor": "loss", "save_top_k": 1, "save_last": False},
            "logging": {"rich_live": False, "log_file": None},
            "export": {"enabled": False},
        },
    )

    run = training_cli.run_fit_command(config_path)
    assert run.result.success is True

    checkpoints = sorted(Path(run.result.run_dir).glob("checkpoints/*.pt"))
    assert checkpoints, "no checkpoint written"
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    # The discriminators travel with the generator, so a resume is not a silent reset.
    assert set(payload["discriminator_optimizer_state_dicts"]) == {"clean", "noise"}
    assert any(k.startswith("disc_clean.") for k in payload["objective_state_dict"])


@pytest.mark.unit
def test_wrapper_factory_must_return_a_trainable_model_wrapper(tmp_path):
    module_name = "cli_training_bad_wrapper"
    (tmp_path / f"{module_name}.py").write_text(
        ADVERSARIAL_MODULE_SOURCE + "\n\ndef build_bad_wrapper(model, **_):\n    return object()\n",
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "wrapper_factory": f"{module_name}:build_bad_wrapper",
                "device": "cpu",
            },
            "data": {"dataset_factory": f"{module_name}:build_dataset"},
            "training": {
                "model_name": "CLIBadWrapper",
                "chunk_size": 12,
                "target_type": "clean",
                "val_ratio": 0.25,
                "max_epochs": 1,
                "batch_size": 4,
                "output_dir": str(tmp_path / "runs"),
            },
            "logging": {"rich_live": False, "log_file": None},
            "export": {"enabled": False},
        },
    )
    pytest.importorskip("torch")
    with pytest.raises(training_cli.ProcessorValidationError, match="must return a TrainableModelWrapper"):
        training_cli.run_fit_command(config_path)


@pytest.mark.unit
def test_loss_factory_receives_run_metadata(tmp_path):
    """A loss that needs ``sfreq`` must actually get it.

    IC-U-Net's frequency term restricts itself to 1-50 Hz and falls back to the
    full spectrum when ``sfreq`` is None. Because the loss factory used to be
    invoked with no injected kwargs, the full-training config optimised the wrong
    band without any warning.
    """
    module_name = "cli_training_loss_metadata"
    (tmp_path / f"{module_name}.py").write_text(
        ADVERSARIAL_MODULE_SOURCE
        + "\n\nSEEN = {}\n\n\ndef build_recording_loss(sfreq=None, target_type=None, n_channels=None, **_):\n"
        "    SEEN['sfreq'] = sfreq\n"
        "    SEEN['target_type'] = target_type\n"
        "    SEEN['n_channels'] = n_channels\n"
        "    import torch\n"
        "    return torch.nn.MSELoss()\n",
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "loss_factory": f"{module_name}:build_recording_loss",
                "wrapper_factory": f"{module_name}:build_wrapper",
                "device": "cpu",
            },
            "data": {"dataset_factory": f"{module_name}:build_dataset"},
            "training": {
                "model_name": "CLILossMetadata",
                "chunk_size": 12,
                "target_type": "clean",
                "val_ratio": 0.25,
                "max_epochs": 1,
                "batch_size": 4,
                "output_dir": str(tmp_path / "runs"),
            },
            "logging": {"rich_live": False, "log_file": None},
            "export": {"enabled": False},
        },
    )
    pytest.importorskip("torch")
    training_cli.run_fit_command(config_path)

    module = importlib.import_module(module_name)
    assert module.SEEN["sfreq"] == pytest.approx(250.0)
    assert module.SEEN["target_type"] == "clean"
    assert module.SEEN["n_channels"] == 2


@pytest.mark.unit
def test_run_summary_records_what_the_model_says_about_itself(tmp_path):
    """Configuration facts that decide how a result may be reported must be in the run dir.

    For an adversarial run that means the per-network parameter counts; for a model
    with a paper and an extension mode it means which one was active. Reading that
    off the operator's memory afterwards is exactly how a extension number ends up
    reported as a paper number.
    """
    pytest.importorskip("torch")
    module_name = "cli_training_summary_describe"
    (tmp_path / f"{module_name}.py").write_text(
        ADVERSARIAL_MODULE_SOURCE + "\n\ndef _describe(self):\n"
        "    return {'n_channels': self.n_channels, 'paper_mode': self.n_channels == 1}\n"
        "\n\nGenerator.describe = _describe\n",
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "wrapper_factory": f"{module_name}:build_wrapper",
                "device": "cpu",
            },
            "data": {"dataset_factory": f"{module_name}:build_dataset"},
            "training": {
                "model_name": "CLISummaryDescribe",
                "chunk_size": 12,
                "target_type": "clean",
                "val_ratio": 0.25,
                "max_epochs": 1,
                "batch_size": 4,
                "output_dir": str(tmp_path / "runs"),
            },
            "logging": {"rich_live": False, "log_file": None},
            "export": {"enabled": False},
        },
    )
    run = training_cli.run_fit_command(config_path)
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))

    assert summary["model"]["n_channels"] == 2
    assert summary["model"]["paper_mode"] is False
    counts = summary["model"]["parameter_counts"]
    assert set(counts) == {"generator", "discriminator_clean", "discriminator_noise"}
    assert all(v > 0 for v in counts.values())


@pytest.mark.unit
def test_run_summary_tolerates_a_model_without_a_describe_hook(tmp_path):
    module_name = "cli_training_summary_plain"
    (tmp_path / f"{module_name}.py").write_text(ADVERSARIAL_MODULE_SOURCE, encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        {
            "model": {
                "framework": "pytorch",
                "factory": f"{module_name}:build_model",
                "wrapper_factory": f"{module_name}:build_wrapper",
                "device": "cpu",
            },
            "data": {"dataset_factory": f"{module_name}:build_dataset"},
            "training": {
                "model_name": "CLISummaryPlain",
                "chunk_size": 12,
                "target_type": "clean",
                "val_ratio": 0.25,
                "max_epochs": 1,
                "batch_size": 4,
                "output_dir": str(tmp_path / "runs"),
            },
            "logging": {"rich_live": False, "log_file": None},
            "export": {"enabled": False},
        },
    )
    pytest.importorskip("torch")
    run = training_cli.run_fit_command(config_path)
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    # No describe() hook, but the wrapper still reports its networks.
    assert set(summary["model"]) == {"parameter_counts"}
