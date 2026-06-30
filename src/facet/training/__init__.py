"""
FACETpy Training Module
=======================

Framework-agnostic infrastructure for training deep-learning models
for EEG artifact correction.

Quick start
-----------
::

    from facet.training import (
        EEGArtifactDataset,
        PyTorchModelWrapper,
        Trainer,
        TrainingConfig,
        CheckpointCallback,
        EarlyStoppingCallback,
        CompositeLoss,
        mse_loss,
        spectral_loss,
    )

    # 1. Build dataset from a FACETpy ProcessingContext
    dataset = EEGArtifactDataset(context, chunk_size=1250)
    train_ds, val_ds = dataset.train_val_split(val_ratio=0.2)

    # 2. Wrap your model
    import torch.nn as nn
    model = nn.Sequential(nn.Conv1d(4, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 4, 3, padding=1))
    wrapper = PyTorchModelWrapper(model=model, loss_fn=nn.MSELoss(), device="cpu")

    # 3. Configure training
    config = TrainingConfig(model_name="MyModel", max_epochs=50, batch_size=16)

    # 4. Train
    result = Trainer(wrapper, train_ds, val_ds, config).fit()
    print(f"Best val_loss {result.best_metric:.4f} @ epoch {result.best_epoch}")
"""

from .callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LossPlotCallback,
    MetricLoggerCallback,
    WandbCallback,
)
from .cli import CLITrainingRun, build_parser, load_training_cli_config, main, run_fit_command
from .config import (
    AugmentationConfig,
    CheckpointConfig,
    EarlyStoppingConfig,
    LoggingConfig,
    TrainingConfig,
)
from .dataset import (
    AmplitudeJitter,
    BackgroundMix,
    ChannelDropout,
    EEGArtifactDataset,
    LengthJitterNoise,
    NoiseScaling,
    NPZContextArtifactDataset,
    NPZSpatioTemporalDataset,
    SignFlip,
    TriggerJitter,
    WindowShift,
)
from .honesty_metrics import honesty_report, honesty_verdict
from .losses import (
    CompositeLoss,
    TorchLossWrapper,
    mae_loss,
    mse_loss,
    snr_loss,
    spectral_loss,
)
from .trainer import Trainer, TrainingResult, TrainingState
from .wrapper import PyTorchModelWrapper, TensorFlowModelWrapper, TrainableModelWrapper

__all__ = [
    # Config
    "TrainingConfig",
    "CheckpointConfig",
    "EarlyStoppingConfig",
    "AugmentationConfig",
    "LoggingConfig",
    # Dataset
    "EEGArtifactDataset",
    "NPZContextArtifactDataset",
    "NPZSpatioTemporalDataset",
    "TriggerJitter",
    "NoiseScaling",
    "ChannelDropout",
    "SignFlip",
    "WindowShift",
    "BackgroundMix",
    "AmplitudeJitter",
    "LengthJitterNoise",
    # Wrappers
    "TrainableModelWrapper",
    "PyTorchModelWrapper",
    "TensorFlowModelWrapper",
    # Losses
    "mse_loss",
    "mae_loss",
    "spectral_loss",
    "snr_loss",
    "CompositeLoss",
    "TorchLossWrapper",
    # Honesty metrics
    "honesty_report",
    "honesty_verdict",
    # Callbacks
    "Callback",
    "CallbackList",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LossPlotCallback",
    "MetricLoggerCallback",
    "WandbCallback",
    # CLI
    "build_parser",
    "load_training_cli_config",
    "run_fit_command",
    "CLITrainingRun",
    "main",
    # Trainer
    "Trainer",
    "TrainingState",
    "TrainingResult",
]
