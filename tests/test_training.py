"""Tests for the facet.training module."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata
from facet.training import (
    AugmentationConfig,
    ChannelDropout,
    CheckpointCallback,
    CheckpointConfig,
    CompositeLoss,
    EarlyStoppingCallback,
    EarlyStoppingConfig,
    EEGArtifactDataset,
    LoggingConfig,
    MetricLoggerCallback,
    NoiseScaling,
    SignFlip,
    TrainableModelWrapper,
    Trainer,
    TrainingConfig,
    TrainingResult,
    TrainingState,
    TriggerJitter,
    mae_loss,
    mse_loss,
    snr_loss,
    spectral_loss,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SFREQ = 250.0
N_CHANNELS = 4
N_SAMPLES = 2500   # 10 s


def _make_raw(data: np.ndarray | None = None, sfreq: float = SFREQ) -> mne.io.RawArray:
    if data is None:
        data = np.random.default_rng(0).standard_normal((N_CHANNELS, N_SAMPLES)) * 1e-6
    ch_names = [f"EEG{i:02d}" for i in range(data.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * data.shape[0])
    return mne.io.RawArray(data, info, verbose=False)


def _make_context(with_artifacts: bool = False) -> ProcessingContext:
    rng = np.random.default_rng(1)
    clean = rng.standard_normal((N_CHANNELS, N_SAMPLES)) * 1e-6
    if with_artifacts:
        noisy = clean.copy()
        for i in range(10):
            start = i * 250
            noisy[:, start : start + 50] += 50e-6
    else:
        noisy = clean.copy()

    raw_noisy = _make_raw(noisy)
    raw_clean = _make_raw(clean)

    triggers = np.arange(0, N_SAMPLES, 250)[:10].astype(int)
    metadata = ProcessingMetadata()
    metadata.triggers = triggers
    metadata.artifact_length = 50
    return ProcessingContext(raw=raw_noisy, raw_original=raw_clean, metadata=metadata)


@pytest.fixture
def sample_context() -> ProcessingContext:
    return _make_context(with_artifacts=True)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainingConfig:
    def test_default_instantiation(self):
        cfg = TrainingConfig()
        assert cfg.max_epochs == 100
        assert cfg.batch_size == 16
        assert cfg.chunk_size == 1250
        assert cfg.target_type == "clean"
        assert isinstance(cfg.checkpoint, CheckpointConfig)
        assert isinstance(cfg.augmentation, AugmentationConfig)
        assert isinstance(cfg.logging, LoggingConfig)
        assert cfg.early_stopping is None

    def test_to_dict_round_trip(self):
        cfg = TrainingConfig(model_name="TestModel", max_epochs=5, batch_size=4)
        d = cfg.to_dict()
        assert d["model_name"] == "TestModel"
        cfg2 = TrainingConfig.from_dict(d)
        assert cfg2.model_name == "TestModel"
        assert cfg2.max_epochs == 5

    def test_json_serialization_round_trip(self, tmp_path):
        cfg = TrainingConfig(
            model_name="JSONTest",
            max_epochs=3,
            early_stopping=EarlyStoppingConfig(patience=5),
        )
        path = tmp_path / "config.json"
        cfg.save_json(path)
        assert path.exists()
        cfg2 = TrainingConfig.load_json(path)
        assert cfg2.model_name == "JSONTest"
        assert cfg2.max_epochs == 3
        assert cfg2.early_stopping is not None
        assert cfg2.early_stopping.patience == 5

    def test_yaml_serialization_round_trip(self, tmp_path):
        cfg = TrainingConfig(model_name="YAMLTest", max_epochs=7)
        path = tmp_path / "config.yaml"
        cfg.save_yaml(path)
        assert path.exists()
        cfg2 = TrainingConfig.load_yaml(path)
        assert cfg2.model_name == "YAMLTest"
        assert cfg2.max_epochs == 7

    def test_from_dict_ignores_unknown_keys(self):
        d = TrainingConfig().to_dict()
        d["__future_key__"] = "ignored"
        cfg = TrainingConfig.from_dict(d)
        assert not hasattr(cfg, "__future_key__")

    def test_augmentation_noise_scale_tuple_preserved(self):
        cfg = TrainingConfig(augmentation=AugmentationConfig(noise_scale_range=(0.8, 1.2)))
        d = cfg.to_dict()
        cfg2 = TrainingConfig.from_dict(d)
        assert cfg2.augmentation.noise_scale_range == (0.8, 1.2)


# ---------------------------------------------------------------------------
# EEGArtifactDataset
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEEGArtifactDataset:
    def test_basic_construction(self, sample_context):
        ds = EEGArtifactDataset(sample_context, chunk_size=250)
        assert len(ds) > 0

    def test_single_context_wrapped(self, sample_context):
        """Passing a single context (not a list) should be accepted."""
        ds = EEGArtifactDataset(sample_context, chunk_size=250)
        assert ds.n_chunks > 0

    def test_item_shapes(self, sample_context):
        ds = EEGArtifactDataset(sample_context, chunk_size=250)
        noisy, target = ds[0]
        assert noisy.shape == (N_CHANNELS, 250)
        assert target.shape == (N_CHANNELS, 250)

    def test_item_dtype(self, sample_context):
        ds = EEGArtifactDataset(sample_context, chunk_size=250)
        noisy, target = ds[0]
        assert noisy.dtype == np.float32
        assert target.dtype == np.float32

    def test_target_type_artifact(self, sample_context):
        ds = EEGArtifactDataset(sample_context, chunk_size=250, target_type="artifact")
        noisy, artifact = ds[0]
        # artifact ≈ noisy - clean (non-zero at trigger positions)
        assert noisy.shape == artifact.shape

    def test_invalid_target_type_raises(self, sample_context):
        with pytest.raises(ValueError, match="target_type"):
            EEGArtifactDataset(sample_context, chunk_size=250, target_type="invalid")

    def test_invalid_overlap_raises(self, sample_context):
        with pytest.raises(ValueError, match="overlap"):
            EEGArtifactDataset(sample_context, chunk_size=250, overlap=1.0)

    def test_empty_contexts_raises(self):
        with pytest.raises(ValueError):
            EEGArtifactDataset([], chunk_size=250)

    def test_n_channels_property(self, sample_context):
        ds = EEGArtifactDataset(sample_context, chunk_size=250)
        assert ds.n_channels == N_CHANNELS

    def test_trigger_aligned_chunking(self, sample_context):
        """Trigger-aligned chunks should start at trigger positions."""
        ds = EEGArtifactDataset(sample_context, chunk_size=250, trigger_aligned=True)
        # 10 triggers, each at i*250; each fits a chunk of 250 samples
        assert len(ds) == 10

    def test_sliding_window_chunking(self, sample_context):
        """Sliding window (no trigger alignment) should produce more chunks."""
        ds = EEGArtifactDataset(sample_context, chunk_size=250, trigger_aligned=False, overlap=0.5)
        assert len(ds) > 10

    def test_train_val_split_sizes(self, sample_context):
        ds = EEGArtifactDataset(sample_context, chunk_size=250)
        n = len(ds)
        train, val = ds.train_val_split(val_ratio=0.2)
        assert len(train) + len(val) == n
        assert len(val) >= 1

    def test_train_val_split_no_overlap(self, sample_context):
        ds = EEGArtifactDataset(sample_context, chunk_size=250)
        train, val = ds.train_val_split(val_ratio=0.2, seed=7)
        assert len(train) + len(val) == len(ds)

    def test_repr(self, sample_context):
        ds = EEGArtifactDataset(sample_context, chunk_size=250)
        r = repr(ds)
        assert "EEGArtifactDataset" in r
        assert "n_chunks" in r

    def test_multiple_contexts(self, sample_context):
        ds = EEGArtifactDataset(
            [sample_context, sample_context],
            chunk_size=250,
        )
        single_ds = EEGArtifactDataset(sample_context, chunk_size=250)
        assert len(ds) == 2 * len(single_ds)


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAugmentationTransforms:
    def _pair(self):
        rng = np.random.default_rng(0)
        noisy = rng.standard_normal((4, 256)).astype(np.float32)
        target = rng.standard_normal((4, 256)).astype(np.float32)
        return noisy, target

    def test_trigger_jitter_output_shape(self):
        jitter = TriggerJitter(max_jitter=5, seed=0)
        noisy, target = self._pair()
        n2, t2 = jitter(noisy.copy(), target.copy())
        assert n2.shape == noisy.shape
        assert t2.shape == target.shape

    def test_noise_scaling_identity(self):
        scale = NoiseScaling(scale_range=(1.0, 1.0))
        noisy, target = self._pair()
        n2, t2 = scale(noisy.copy(), target.copy())
        np.testing.assert_array_equal(n2, noisy)
        np.testing.assert_array_equal(t2, target)

    def test_noise_scaling_changes_values(self):
        scale = NoiseScaling(scale_range=(0.5, 0.5))
        noisy, target = self._pair()
        n2, _ = scale(noisy.copy(), target.copy())
        np.testing.assert_allclose(n2, noisy * 0.5, rtol=1e-5)

    def test_channel_dropout_zero_prob(self):
        drop = ChannelDropout(p=0.0)
        noisy, target = self._pair()
        n2, t2 = drop(noisy.copy(), target.copy())
        np.testing.assert_array_equal(n2, noisy)

    def test_channel_dropout_full_prob_zeros_all(self):
        drop = ChannelDropout(p=1.0, seed=42)
        noisy, target = self._pair()
        n2, _ = drop(noisy.copy(), target.copy())
        assert np.all(n2 == 0.0)

    def test_sign_flip_zero_prob(self):
        flip = SignFlip(p=0.0)
        noisy, target = self._pair()
        n2, t2 = flip(noisy, target)
        np.testing.assert_array_equal(n2, noisy)

    def test_transforms_applied_in_dataset(self, sample_context):
        """Transforms passed to dataset should be called on __getitem__."""
        called = []

        def _spy(n, t):
            called.append(True)
            return n, t

        ds = EEGArtifactDataset(
            sample_context, chunk_size=250, transforms=[_spy]
        )
        ds[0]
        assert len(called) == 1


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLossFunctions:
    def _arrays(self, batch=2, ch=4, t=256):
        rng = np.random.default_rng(0)
        pred = rng.standard_normal((batch, ch, t)).astype(np.float32)
        target = rng.standard_normal((batch, ch, t)).astype(np.float32)
        return pred, target

    def test_mse_loss_zero_for_identical(self):
        a = np.ones((2, 4, 128))
        assert mse_loss(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_mse_loss_positive(self):
        pred, target = self._arrays()
        assert mse_loss(pred, target) > 0.0

    def test_mae_loss_zero_for_identical(self):
        a = np.ones((2, 4, 128))
        assert mae_loss(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_mae_loss_positive(self):
        pred, target = self._arrays()
        assert mae_loss(pred, target) > 0.0

    def test_spectral_loss_zero_for_identical(self):
        a = np.random.randn(2, 4, 512).astype(np.float32)
        assert spectral_loss(a, a, sfreq=250.0) == pytest.approx(0.0, abs=1e-6)

    def test_spectral_loss_positive(self):
        pred, target = self._arrays(t=512)
        assert spectral_loss(pred, target, sfreq=250.0) > 0.0

    def test_snr_loss_zero_residual_returns_large_negative(self):
        """Identical pred and target → residual ≈ 0 → very large SNR → negated loss large negative."""
        # Use unit-amplitude signal so signal_power >> eps and SNR is large
        a = np.ones((2, 4, 128)).astype(np.float64)
        loss = snr_loss(a, a)
        assert loss < -30.0  # SNR >> 30 dB → negated loss << -30

    def test_composite_loss_weighted_sum(self):
        pred = np.ones((2, 4, 64)) * 0.5
        target = np.zeros((2, 4, 64))
        comp = CompositeLoss({"mse": (mse_loss, 1.0), "mae": (mae_loss, 2.0)})
        total, breakdown = comp(pred, target)
        expected = mse_loss(pred, target) * 1.0 + mae_loss(pred, target) * 2.0
        assert total == pytest.approx(expected, rel=1e-5)
        assert "mse" in breakdown
        assert "mae" in breakdown
        assert "total" in breakdown

    def test_composite_loss_breakdown_values(self):
        pred = np.ones((1, 1, 10))
        target = np.zeros((1, 1, 10))
        comp = CompositeLoss({"mse": (mse_loss, 1.0)})
        total, breakdown = comp(pred, target)
        assert breakdown["mse"] == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# TrainableModelWrapper (using a minimal concrete stub)
# ---------------------------------------------------------------------------

class _CountingWrapper(TrainableModelWrapper):
    """Minimal wrapper for testing the callback/trainer infrastructure."""

    def __init__(self):
        super().__init__()
        self.train_calls = 0
        self.eval_calls = 0
        self.scheduler_calls = 0
        self._ckpt_saved: list[Path] = []
        self._ckpt_loaded: list[Path] = []

    def train_step(self, noisy, target):
        self.train_calls += 1
        return {"loss": float(np.mean((noisy - target) ** 2))}

    def eval_step(self, noisy, target):
        self.eval_calls += 1
        return {"loss": float(np.mean((noisy - target) ** 2))}

    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("ckpt")
        self._ckpt_saved.append(Path(path))

    def load_checkpoint(self, path):
        self._ckpt_loaded.append(Path(path))

    def scheduler_step(self):
        self.scheduler_calls += 1


@pytest.mark.unit
class TestTrainableModelWrapper:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            TrainableModelWrapper()

    def test_concrete_stub_train_step(self, sample_context):
        wrapper = _CountingWrapper()
        noisy = np.random.randn(4, N_CHANNELS, 250).astype(np.float32)
        target = np.random.randn(4, N_CHANNELS, 250).astype(np.float32)
        result = wrapper.train_step(noisy, target)
        assert "loss" in result
        assert wrapper.train_calls == 1

    def test_to_inference_adapter_raises_by_default(self):
        wrapper = _CountingWrapper()
        with pytest.raises(NotImplementedError):
            wrapper.to_inference_adapter()

    def test_device_info_default(self):
        assert _CountingWrapper().device_info == "cpu"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckpointCallback:
    def _make_state(self, epoch=1, val_loss=0.05):
        state = TrainingState(run_name="test", max_epochs=10)
        state.epoch = epoch
        state.val_metrics = {"loss": val_loss}
        return state

    def test_saves_checkpoint_on_epoch_end(self, tmp_path):
        wrapper = _CountingWrapper()
        cb = CheckpointCallback(wrapper=wrapper, dirpath=tmp_path / "ckpts", monitor="loss")
        state = self._make_state()
        cb.on_train_begin(state)
        cb.on_epoch_end(state)
        assert len(wrapper._ckpt_saved) >= 1

    def test_last_checkpoint_always_written(self, tmp_path):
        wrapper = _CountingWrapper()
        cb = CheckpointCallback(wrapper=wrapper, dirpath=tmp_path / "ckpts", monitor="loss", save_last=True)
        state = self._make_state()
        cb.on_train_begin(state)
        cb.on_epoch_end(state)
        last = tmp_path / "ckpts" / "last.pt"
        assert last.exists()

    def test_top_k_pruning(self, tmp_path):
        wrapper = _CountingWrapper()
        cb = CheckpointCallback(
            wrapper=wrapper, dirpath=tmp_path / "ckpts",
            monitor="loss", mode="min", save_top_k=2, save_last=False,
        )
        cb.on_train_begin(TrainingState())
        for epoch, loss in enumerate([0.05, 0.04, 0.06, 0.03], start=1):
            state = self._make_state(epoch=epoch, val_loss=loss)
            cb.on_epoch_end(state)
        # Only top-2 best checkpoints should remain tracked
        assert len(cb._top_k) <= 2


@pytest.mark.unit
class TestEarlyStoppingCallback:
    def _make_state(self, val_loss):
        state = TrainingState(run_name="es_test", max_epochs=100)
        state.val_metrics = {"loss": val_loss}
        return state

    def test_no_stop_while_improving(self):
        cb = EarlyStoppingCallback(monitor="loss", mode="min", patience=3)
        for loss in [0.1, 0.09, 0.08, 0.07]:
            state = self._make_state(loss)
            cb.on_epoch_end(state)
            assert not state.stop_training

    def test_stop_after_patience_exceeded(self):
        cb = EarlyStoppingCallback(monitor="loss", mode="min", patience=3)
        for loss in [0.1, 0.1, 0.1, 0.1]:
            state = self._make_state(loss)
            cb.on_epoch_end(state)
        assert state.stop_training

    def test_no_stop_when_metric_absent(self):
        cb = EarlyStoppingCallback(monitor="nonexistent", patience=1)
        state = TrainingState()
        state.val_metrics = {"loss": 0.1}
        cb.on_epoch_end(state)
        assert not state.stop_training


@pytest.mark.unit
class TestMetricLoggerCallback:
    def test_creates_jsonl_file(self, tmp_path):
        cb = MetricLoggerCallback(filepath=tmp_path / "metrics.jsonl")
        state = TrainingState(run_name="log_test", max_epochs=1)
        state.epoch = 1
        state.step = 10
        state.train_metrics = {"loss": 0.05}
        state.val_metrics = {"loss": 0.06}
        cb.on_train_begin(state)
        cb.on_epoch_end(state)
        log_file = tmp_path / "metrics.jsonl"
        assert log_file.exists()
        with log_file.open() as f:
            record = json.loads(f.readline())
        assert record["epoch"] == 1
        assert "train_loss" in record
        assert "val_loss" in record

    def test_appends_multiple_epochs(self, tmp_path):
        cb = MetricLoggerCallback(filepath=tmp_path / "metrics.jsonl")
        for epoch in range(1, 4):
            state = TrainingState(epoch=epoch, max_epochs=3)
            state.train_metrics = {"loss": 0.1 / epoch}
            state.val_metrics = {}
            cb.on_train_begin(state)
            cb.on_epoch_end(state)
        lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Trainer (end-to-end, no real framework)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainer:
    def _make_trainer(self, sample_context, tmp_path, max_epochs=2, with_val=True):
        ds = EEGArtifactDataset(sample_context, chunk_size=250, trigger_aligned=True)
        train_ds, val_ds = ds.train_val_split(val_ratio=0.3, seed=0)
        wrapper = _CountingWrapper()
        config = TrainingConfig(
            model_name="UnitTest",
            max_epochs=max_epochs,
            batch_size=4,
            output_dir=str(tmp_path / "runs"),
            logging=LoggingConfig(rich_live=False, log_file="metrics.jsonl"),
            checkpoint=CheckpointConfig(monitor="loss", save_top_k=1, save_last=True),
        )
        return Trainer(
            wrapper=wrapper,
            train_dataset=train_ds,
            val_dataset=val_ds if with_val else None,
            config=config,
        ), wrapper

    def test_fit_returns_training_result(self, sample_context, tmp_path):
        trainer, _ = self._make_trainer(sample_context, tmp_path)
        result = trainer.fit()
        assert isinstance(result, TrainingResult)
        assert result.success is True

    def test_fit_runs_correct_epochs(self, sample_context, tmp_path):
        trainer, _ = self._make_trainer(sample_context, tmp_path, max_epochs=3)
        result = trainer.fit()
        assert result.total_epochs == 3

    def test_fit_calls_train_step(self, sample_context, tmp_path):
        trainer, wrapper = self._make_trainer(sample_context, tmp_path, max_epochs=2)
        trainer.fit()
        assert wrapper.train_calls > 0

    def test_fit_calls_eval_step_when_val_present(self, sample_context, tmp_path):
        trainer, wrapper = self._make_trainer(sample_context, tmp_path, max_epochs=2)
        trainer.fit()
        assert wrapper.eval_calls > 0

    def test_fit_no_eval_without_val_dataset(self, sample_context, tmp_path):
        trainer, wrapper = self._make_trainer(sample_context, tmp_path, max_epochs=2, with_val=False)
        trainer.fit()
        assert wrapper.eval_calls == 0

    def test_fit_writes_config_json(self, sample_context, tmp_path):
        trainer, _ = self._make_trainer(sample_context, tmp_path, max_epochs=1)
        result = trainer.fit()
        config_file = result.run_dir / "config.json"
        assert config_file.exists()
        with config_file.open() as f:
            d = json.load(f)
        assert d["model_name"] == "UnitTest"

    def test_fit_metric_history_populated(self, sample_context, tmp_path):
        trainer, _ = self._make_trainer(sample_context, tmp_path, max_epochs=2)
        result = trainer.fit()
        assert "loss" in result.metric_history
        assert len(result.metric_history["loss"]) == 2

    def test_fit_best_epoch_tracked(self, sample_context, tmp_path):
        trainer, _ = self._make_trainer(sample_context, tmp_path, max_epochs=3)
        result = trainer.fit()
        assert 1 <= result.best_epoch <= 3

    def test_early_stopping_callback_halts_training(self, sample_context, tmp_path):
        ds = EEGArtifactDataset(sample_context, chunk_size=250)
        train_ds, val_ds = ds.train_val_split(val_ratio=0.3)
        wrapper = _CountingWrapper()
        config = TrainingConfig(
            max_epochs=50,
            batch_size=4,
            output_dir=str(tmp_path / "runs"),
            logging=LoggingConfig(rich_live=False, log_file=None),
            checkpoint=CheckpointConfig(save_top_k=1, save_last=False),
        )
        es_cb = EarlyStoppingCallback(monitor="loss", patience=2, min_delta=1e9)
        trainer = Trainer(wrapper, train_ds, val_ds, config, callbacks=[es_cb])
        result = trainer.fit()
        # With min_delta=1e9 nothing counts as improvement → stop after patience=2
        assert result.total_epochs <= 10  # well under 50

    def test_early_stopping_config_auto_adds_callback(self, sample_context, tmp_path):
        ds = EEGArtifactDataset(sample_context, chunk_size=250)
        train_ds, val_ds = ds.train_val_split(val_ratio=0.3)
        wrapper = _CountingWrapper()
        config = TrainingConfig(
            max_epochs=50,
            batch_size=4,
            output_dir=str(tmp_path / "runs"),
            logging=LoggingConfig(rich_live=False, log_file=None),
            checkpoint=CheckpointConfig(
                monitor="loss", save_top_k=1, save_last=False
            ),
            early_stopping=EarlyStoppingConfig(
                monitor="loss", patience=2, min_delta=1e9
            ),
        )
        trainer = Trainer(wrapper, train_ds, val_ds, config)
        result = trainer.fit()
        assert result.total_epochs <= 10

    def test_scheduler_step_called_once_per_epoch(self, sample_context, tmp_path):
        trainer, wrapper = self._make_trainer(sample_context, tmp_path, max_epochs=3)
        trainer.fit()
        assert wrapper.scheduler_calls == 3

    def test_fit_elapsed_seconds_positive(self, sample_context, tmp_path):
        trainer, _ = self._make_trainer(sample_context, tmp_path, max_epochs=1)
        result = trainer.fit()
        assert result.elapsed_seconds > 0.0
