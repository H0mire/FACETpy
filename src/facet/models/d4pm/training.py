"""Training factories for the D4PM single-branch conditional diffusion model.

The reference paper (arXiv 2509.14302) trains two independent diffusion
branches (clean EEG, artifact). Our adaptation trains only the artifact
branch as a conditional diffusion model p(h | y), using the supervised
``artifact_center`` and ``noisy_center`` arrays from the Niazy proof-fit
dataset. See documentation/research_notes.md for the rationale.

The exposed factories follow the FACETpy training CLI contract
(``build_model``, ``build_loss``, ``build_dataset``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_NOISY_KEY = "noisy_center"
_ARTIFACT_KEY = "artifact_center"


def make_beta_schedule(num_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Linear beta schedule used by the reference D4PM code."""
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)


class SinusoidalNoiseLevelEmbedding(nn.Module):
    """Map a continuous noise level to a sinusoidal embedding.

    Mirrors the reference D4PM PositionalEncoding which encodes
    ``noise_level`` (a scalar in [0, 1]) instead of an integer timestep.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")
        self.dim = int(dim)

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = noise_level.device
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = noise_level.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation conditioned on noise embedding."""

    def __init__(self, embed_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(embed_dim, 2 * feature_dim)
        self.feature_dim = int(feature_dim)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        scale_shift = self.to_scale_shift(embed)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


class TransformerEncoderLayer1D(nn.Module):
    """Self-attention + FFN block operating on (B, T, C) tensors."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class D4PMNoisePredictor(nn.Module):
    """Transformer-style ε predictor with FiLM noise-level conditioning.

    Implements the dual-stream EEG-DNet from the reference D4PM repository
    in a slimmed-down form. Two parallel streams process the noisy state
    ``h_t`` and the conditioning observation ``y`` respectively. Their
    activations are added after each block and modulated by FiLM blocks
    driven by the diffusion noise-level embedding.
    """

    def __init__(
        self,
        feats: int = 64,
        d_model: int = 128,
        d_ff: int = 512,
        n_heads: int = 2,
        n_layers: int = 2,
        embed_dim: int = 128,
        epoch_samples: int = 512,
    ) -> None:
        super().__init__()
        self.feats = int(feats)
        self.d_model = int(d_model)
        self.epoch_samples = int(epoch_samples)

        self.embed = SinusoidalNoiseLevelEmbedding(embed_dim)
        self.embed_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.in_x = nn.Conv1d(1, feats, kernel_size=3, padding=1)
        self.in_cond = nn.Conv1d(1, feats, kernel_size=3, padding=1)

        self.proj_in = nn.Linear(feats, d_model)

        self.layers_x = nn.ModuleList(
            [TransformerEncoderLayer1D(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.layers_cond = nn.ModuleList(
            [TransformerEncoderLayer1D(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.films = nn.ModuleList(
            [FiLM(embed_dim=embed_dim, feature_dim=d_model) for _ in range(n_layers)]
        )

        self.proj_out = nn.Linear(d_model, feats)
        self.out_conv = nn.Sequential(
            nn.Conv1d(feats, feats, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(feats, 1, kernel_size=3, padding=1),
        )

    def forward(
        self, h_t: torch.Tensor, cond_y: torch.Tensor, noise_level: torch.Tensor
    ) -> torch.Tensor:
        embed = self.embed_mlp(self.embed(noise_level))

        x = self.in_x(h_t)
        c = self.in_cond(cond_y)
        x = self.proj_in(x.transpose(1, 2))
        c = self.proj_in(c.transpose(1, 2))

        for layer_x, layer_c, film in zip(self.layers_x, self.layers_cond, self.films):
            x = layer_x(x)
            c = layer_c(c)
            x = x + c
            x = film(x.transpose(1, 2), embed).transpose(1, 2)

        x = self.proj_out(x).transpose(1, 2)
        return self.out_conv(x)


class D4PMTrainingModule(nn.Module):
    """Wraps the noise predictor with the diffusion training-step logic.

    The FACETpy training CLI calls ``model(packed_input)`` followed by
    ``loss_fn(prediction, target)``. To stay inside that contract while
    still implementing diffusion training, the dataset packs both the
    conditioning observation ``y`` and the target artifact ``h0`` into a
    single tensor of shape ``(B, 2, T)``. This module unpacks them,
    samples a random noise level and Gaussian noise ``ε``, forms ``h_t``
    via the forward diffusion equation, and returns
    ``(pred_noise, true_noise)`` stacked along channel dim. The custom
    loss is then ``MSE/L1(pred_noise, true_noise)``.
    """

    def __init__(
        self,
        epoch_samples: int = 512,
        num_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        feats: int = 64,
        d_model: int = 128,
        d_ff: int = 512,
        n_heads: int = 2,
        n_layers: int = 2,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.epoch_samples = int(epoch_samples)
        self.num_steps = int(num_steps)

        betas = make_beta_schedule(num_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]]
        )

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).float(),
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance).float())
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float(),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float(),
        )

        self.predictor = D4PMNoisePredictor(
            feats=feats,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            embed_dim=embed_dim,
            epoch_samples=epoch_samples,
        )

    def q_sample(
        self, h0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * h0 + sqrt_one_minus * noise

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.shape[1] != 2:
            raise ValueError(
                f"D4PMTrainingModule expects packed input with 2 channels, got {packed.shape[1]}"
            )
        cond_y = packed[:, 0:1, :]
        h0 = packed[:, 1:2, :]
        batch_size = packed.shape[0]
        device = packed.device

        t = torch.randint(0, self.num_steps, (batch_size,), device=device)
        noise = torch.randn_like(h0)
        h_t = self.q_sample(h0, t, noise)

        noise_level = self.sqrt_alphas_cumprod[t]
        pred_noise = self.predictor(h_t, cond_y, noise_level)

        return torch.cat([pred_noise, noise], dim=1)


class D4PMArtifactDataset:
    """Dataset that packs (noisy_center, artifact_center) per channel.

    Each example produces:

    - input: ``(2, samples)`` with row 0 = noisy_y, row 1 = target artifact
    - target: ``(1, samples)`` zeros (ignored by the diffusion loss)

    Channels are flattened so the checkpoint is channel-count independent.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        demean_input: bool = True,
        demean_target: bool = True,
        max_examples: int | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)

        with np.load(self.path, allow_pickle=True) as bundle:
            self.noisy = bundle[_NOISY_KEY].astype(np.float32, copy=False)
            self.artifact = bundle[_ARTIFACT_KEY].astype(np.float32, copy=False)
            self.sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

        if self.noisy.ndim != 3 or self.artifact.ndim != 3:
            raise ValueError(
                f"D4PM dataset expects 3D arrays (examples, channels, samples), "
                f"got noisy={self.noisy.shape}, artifact={self.artifact.shape}"
            )
        if self.noisy.shape != self.artifact.shape:
            raise ValueError(
                f"noisy_center and artifact_center must match; got "
                f"{self.noisy.shape} vs {self.artifact.shape}"
            )

        self.n_examples = int(self.noisy.shape[0])
        self.n_channels = int(self.noisy.shape[1])
        self.epoch_samples = int(self.noisy.shape[2])
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True

        total = self.n_examples * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        base_idx = int(idx) // self.n_channels
        ch_idx = int(idx) % self.n_channels
        noisy_row = self.noisy[base_idx, ch_idx : ch_idx + 1].astype(np.float32, copy=True)
        artifact_row = self.artifact[base_idx, ch_idx : ch_idx + 1].astype(np.float32, copy=True)
        if self.demean_input:
            noisy_row -= noisy_row.mean(axis=-1, keepdims=True)
        if self.demean_target:
            artifact_row -= artifact_row.mean(axis=-1, keepdims=True)
        packed = np.concatenate([noisy_row, artifact_row], axis=0)
        dummy_target = np.zeros((1, self.epoch_samples), dtype=np.float32)
        return packed, dummy_target

    @property
    def input_shape(self) -> tuple[int, int]:
        return (2, self.epoch_samples)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (1, self.epoch_samples)

    @property
    def n_chunks(self) -> int:
        return len(self)

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42):
        n = len(self)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n).tolist()
        n_val = max(1, int(n * val_ratio))
        val_idx = set(indices[:n_val])
        train_indices = [i for i in range(n) if i not in val_idx]
        val_indices = [i for i in range(n) if i in val_idx]
        return _Subset(self, train_indices), _Subset(self, val_indices)


class _Subset:
    def __init__(self, parent: D4PMArtifactDataset, indices: list[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


class D4PMEpsilonLoss(nn.Module):
    """L1 loss on (predicted ε, true ε) packed in the model output."""

    def __init__(self, kind: str = "l1") -> None:
        super().__init__()
        kind = kind.strip().lower()
        if kind == "l1":
            self.loss_fn = nn.L1Loss()
        elif kind in {"l2", "mse"}:
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported D4PM loss kind: {kind}")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        if prediction.shape[1] != 2:
            raise ValueError(
                f"D4PMEpsilonLoss expects (B, 2, T) prediction, got {prediction.shape}"
            )
        pred_noise = prediction[:, 0:1, :]
        true_noise = prediction[:, 1:2, :]
        return self.loss_fn(pred_noise, true_noise)


def build_model(
    epoch_samples: int | None = 512,
    num_steps: int = 200,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    feats: int = 64,
    d_model: int = 128,
    d_ff: int = 512,
    n_heads: int = 2,
    n_layers: int = 2,
    embed_dim: int = 128,
    input_shape: tuple[int, ...] | None = None,
    **_: object,
) -> D4PMTrainingModule:
    """CLI entry point for the model factory."""
    samples = epoch_samples
    if samples is None and input_shape is not None:
        samples = int(input_shape[-1])
    if samples is None:
        raise ValueError("build_model requires epoch_samples or input_shape")
    return D4PMTrainingModule(
        epoch_samples=int(samples),
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        feats=feats,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        n_layers=n_layers,
        embed_dim=embed_dim,
    )


def build_loss(kind: str = "l1") -> D4PMEpsilonLoss:
    return D4PMEpsilonLoss(kind=kind)


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> D4PMArtifactDataset:
    dataset_path = path or context_path
    if not dataset_path:
        raise ValueError("build_dataset requires path or context_path")
    return D4PMArtifactDataset(
        path=dataset_path,
        demean_input=demean_input,
        demean_target=demean_target,
        max_examples=max_examples,
    )
