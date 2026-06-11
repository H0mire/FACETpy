"""Paper-accurate training factories for the D4PM diffusion model.

This is the *paper-accurate edition* of the D4PM gradient-artifact predictor.
It is a separate model package from ``facet.models.d4pm`` and brings the
implementation closer to the source paper:

    Shao et al., "A Dual-Branch Driven Denoising Diffusion Probabilistic
    Model with Joint Posterior Diffusion Sampling for EEG Artifacts
    Removal", arXiv:2509.14302. Reference code:
    https://github.com/flysnow1024/D4PM (denoising_model_eegdnet_class.py).

What changed vs. the original single-branch ``d4pm`` (see README.md and
documentation/paper_accuracy_review.md for the full discrepancy table):

1. CONTINUOUS noise-level conditioning (Section 2.1). During training we draw
   an integer step ``t`` and then sample a continuous noise level
   ``sqrt(abar*_t) ~ U(sqrt(abar_{t-1}), sqrt(abar_t))``, used for BOTH the
   forward (``q_sample``) corruption AND the conditioning embedding. The
   original used the discrete ``sqrt(abar_t)`` only. This is the paper's
   explicit stabilization trick.
2. THREE Transformer encoder blocks per path by default (Fig 2 "x3"). The
   original defaulted to 2.
3. A Dual-FiLM module that embeds BOTH the noise level AND a categorical
   artifact-class label ``z`` (Section 2.2). With the FACETpy gradient task
   there is a single class (index 0), so the class pathway acts as a learned
   global bias, but it is wired so multi-artifact training is a config change.
4. The Dual-FiLM is applied symmetrically INSIDE each path before fusion, and
   the output head adds a 1x1 fusion conv after the 3x1 conv (paper's
   "(3x1, 1x1)" projection).
5. An OPTIONAL second (clean/EEG) branch (``dual_branch=True``) supervised
   from ``clean_center``, enabling the paper's Joint Posterior Sampling at
   inference (implemented in processor.py). The original was single-branch.
6. An explicit ``lambda_snr`` parameter for the Eq-2 mixture
   ``y = x + x'*lambda_snr`` (default 1.0 because the Niazy NPZ already bakes
   in the recorded mixture, so we do not re-mix).
7. Post-norm Transformer blocks are kept (faithful to the reference EEG-DNet).
8. L1 epsilon-prediction loss is kept as default (Eq 1).

The factories follow the FACETpy training CLI contract
(``build_model``, ``build_loss``, ``build_dataset``) and run on CPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_NOISY_KEY = "noisy_center"
_ARTIFACT_KEY = "artifact_center"
_CLEAN_KEY = "clean_center"


def make_beta_schedule(num_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Linear beta schedule used by the reference D4PM code."""
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)


class SinusoidalNoiseLevelEmbedding(nn.Module):
    """Map a continuous noise level to a sinusoidal embedding.

    Mirrors the reference D4PM PositionalEncoding which encodes the
    continuous ``noise_level`` (an alpha-bar value in [0, 1]) instead of an
    integer timestep. Faithful to the paper's continuous-time conditioning.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")
        self.dim = int(dim)

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = noise_level.device
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half)
        args = noise_level.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ClassEmbedding(nn.Module):
    """Learned embedding for the categorical artifact-class label ``z``.

    Paper (Section 2.2): the Dual-FiLM conditions on BOTH the noise level and
    a categorical class label z. For FACETpy gradient-artifact removal there
    is a single artifact class, so the default ``num_classes=1`` collapses
    this to a learned global bias term; the pathway is kept so multi-artifact
    (gradient + BCG) training only needs ``num_classes>1`` and a class index
    in the dataset (documented deviation).
    """

    def __init__(self, num_classes: int, embed_dim: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embed = nn.Embedding(int(num_classes), int(embed_dim))

    def forward(self, class_idx: torch.Tensor) -> torch.Tensor:
        return self.embed(class_idx.long())


class DualFiLM(nn.Module):
    """Shared Dual-FiLM: channel-wise (gamma, xi) from noise + class embeddings.

    Paper (Section 2.2): a shared module embeds the noise level and the class
    label z into channel-wise scale (gamma) and shift (xi) parameters applied
    within each path. We add the noise embedding and class embedding, then
    project to ``2 * feature_dim`` scale/shift values applied as
    ``x * (1 + gamma) + xi`` along the channel dimension of a ``(B, C, T)``
    tensor.
    """

    def __init__(self, embed_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.to_scale_shift = nn.Linear(int(embed_dim), 2 * int(feature_dim))

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        scale_shift = self.to_scale_shift(cond_embed)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


class TransformerEncoderLayer1D(nn.Module):
    """Post-norm self-attention + FFN block operating on (B, T, C) tensors.

    Faithful to the reference EEG-DNet EncoderLayer (classic post-norm
    Add&Norm). ``norm_first`` is exposed but defaults to ``False`` (post-norm)
    to match the reference.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.norm_first = bool(norm_first)
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
        if self.norm_first:
            n = self.norm1(x)
            attn_out, _ = self.attn(n, n, n, need_weights=False)
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
            return x
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class D4PMNoisePredictor(nn.Module):
    """Paper-accurate dual-path epsilon predictor with shared Dual-FiLM.

    Two identical paths process the noise-perturbed state ``h_t`` and the
    conditioning observation ``y``. Each path = Conv1d(1->feats, k3) ->
    Linear(feats->d_model) -> ``n_layers`` post-norm Transformer encoder
    blocks (default 3, the paper's "x3"). A single SHARED Dual-FiLM
    (conditioned on noise level + class label z) modulates each path
    symmetrically before fusion. The two paths are added, projected back to
    ``feats``, and passed through the (3x1, 1x1) output head to predict epsilon.
    """

    def __init__(
        self,
        feats: int = 64,
        d_model: int = 128,
        d_ff: int = 512,
        n_heads: int = 2,
        n_layers: int = 3,
        embed_dim: int = 128,
        num_classes: int = 1,
        norm_first: bool = False,
        epoch_samples: int = 512,
    ) -> None:
        super().__init__()
        self.feats = int(feats)
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.epoch_samples = int(epoch_samples)

        self.noise_embed = SinusoidalNoiseLevelEmbedding(embed_dim)
        self.embed_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.class_embed = ClassEmbedding(num_classes=num_classes, embed_dim=embed_dim)

        self.in_x = nn.Conv1d(1, feats, kernel_size=3, padding=1)
        self.in_cond = nn.Conv1d(1, feats, kernel_size=3, padding=1)

        self.proj_in = nn.Linear(feats, d_model)

        self.layers_x = nn.ModuleList(
            [TransformerEncoderLayer1D(d_model, n_heads, d_ff, norm_first=norm_first) for _ in range(n_layers)]
        )
        self.layers_cond = nn.ModuleList(
            [TransformerEncoderLayer1D(d_model, n_heads, d_ff, norm_first=norm_first) for _ in range(n_layers)]
        )
        # ONE shared Dual-FiLM per depth level, applied symmetrically to both
        # paths (paper: "shared Dual-FiLM").
        self.films = nn.ModuleList([DualFiLM(embed_dim=embed_dim, feature_dim=d_model) for _ in range(n_layers)])

        self.proj_out = nn.Linear(d_model, feats)
        # Paper's (3x1, 1x1) projection head.
        self.out_conv3 = nn.Conv1d(feats, feats, kernel_size=3, padding=1)
        self.out_act = nn.ReLU()
        self.out_conv1 = nn.Conv1d(feats, 1, kernel_size=1)

    def forward(
        self,
        h_t: torch.Tensor,
        cond_y: torch.Tensor,
        noise_level: torch.Tensor,
        class_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = h_t.shape[0]
        device = h_t.device
        if class_idx is None:
            class_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

        noise_e = self.embed_mlp(self.noise_embed(noise_level))
        class_e = self.class_embed(class_idx)
        cond_embed = noise_e + class_e

        x = self.in_x(h_t)
        c = self.in_cond(cond_y)
        x = self.proj_in(x.transpose(1, 2))
        c = self.proj_in(c.transpose(1, 2))

        for layer_x, layer_c, film in zip(self.layers_x, self.layers_cond, self.films, strict=False):
            x = layer_x(x)
            c = layer_c(c)
            # Shared Dual-FiLM applied symmetrically to BOTH paths before fusion.
            x = film(x.transpose(1, 2), cond_embed).transpose(1, 2)
            c = film(c.transpose(1, 2), cond_embed).transpose(1, 2)
            x = x + c

        x = self.proj_out(x).transpose(1, 2)
        x = self.out_act(self.out_conv3(x))
        return self.out_conv1(x)


class D4PMTrainingModule(nn.Module):
    """Wraps the noise predictor(s) with diffusion training-step logic.

    Single-branch (``dual_branch=False``): the dataset packs ``(y, artifact0)``
    into ``(B, 2, T)``; this module samples a CONTINUOUS noise level and
    Gaussian noise epsilon, forms ``h_t = q_sample`` with that continuous
    level, predicts epsilon, and returns ``cat([pred_eps, true_eps])`` as
    ``(B, 2, T)``.

    Dual-branch (``dual_branch=True``, the paper-faithful default): the dataset
    packs ``(y, artifact0, clean0)`` into ``(B, 3, T)``; this module holds TWO
    predictors (artifact + clean), samples a SHARED continuous level and a
    SHARED epsilon, predicts epsilon for both branches, and returns
    ``cat([pred_eps_art, true_eps_art, pred_eps_clean, true_eps_clean])`` as
    ``(B, 4, T)``.

    The custom loss is L1/L2 between each ``(pred, true)`` epsilon pair.
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
        n_layers: int = 3,
        embed_dim: int = 128,
        num_classes: int = 1,
        norm_first: bool = False,
        dual_branch: bool = True,
        lambda_snr: float = 1.0,
    ) -> None:
        super().__init__()
        self.epoch_samples = int(epoch_samples)
        self.num_steps = int(num_steps)
        self.dual_branch = bool(dual_branch)
        self.lambda_snr = float(lambda_snr)

        betas = make_beta_schedule(num_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).float(),
        )
        # sqrt(abar) at the previous step (1.0 at t=0) -> lower bound of the
        # continuous-level interval [sqrt(abar_{t-1}), sqrt(abar_t)].
        self.register_buffer("sqrt_alphas_cumprod_prev", torch.sqrt(alphas_cumprod_prev).float())
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

        predictor_kwargs = dict(
            feats=feats,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            embed_dim=embed_dim,
            num_classes=num_classes,
            norm_first=norm_first,
            epoch_samples=epoch_samples,
        )
        # Artifact branch (always present).
        self.predictor = D4PMNoisePredictor(**predictor_kwargs)
        # Optional EEG/clean branch for joint posterior sampling. Independent
        # weights -> no sharing (paper Fig 2).
        self.predictor_clean = D4PMNoisePredictor(**predictor_kwargs) if self.dual_branch else None

    def continuous_noise_level(self, t: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample sqrt(abar*_t) uniformly in [sqrt(abar_{t-1}), sqrt(abar_t)].

        ``u`` is a uniform sample in [0, 1) (per example). Returns the
        continuous ``sqrt(abar*)`` and the matching ``sqrt(1 - abar*)``.
        """
        lo = self.sqrt_alphas_cumprod_prev[t]
        hi = self.sqrt_alphas_cumprod[t]
        sqrt_abar = lo + u * (hi - lo)
        sqrt_abar = torch.clamp(sqrt_abar, max=1.0)
        sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - sqrt_abar * sqrt_abar, min=0.0))
        return sqrt_abar, sqrt_one_minus

    def q_sample_continuous(
        self, x0: torch.Tensor, sqrt_abar: torch.Tensor, sqrt_one_minus: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_abar = sqrt_abar.view(-1, 1, 1)
        sqrt_one_minus = sqrt_one_minus.view(-1, 1, 1)
        return sqrt_abar * x0 + sqrt_one_minus * noise

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        expected = 3 if self.dual_branch else 2
        if packed.shape[1] != expected:
            raise ValueError(
                f"D4PMTrainingModule(dual_branch={self.dual_branch}) expects packed input with "
                f"{expected} channels, got {packed.shape[1]}"
            )
        if torch.jit.is_tracing():
            # Trace-stable stub for the CLI's torch.jit.trace export. Real
            # inference uses the state-dict checkpoint via the adapter's
            # iterative sampler. MultiheadAttention kernel-selection
            # non-determinism otherwise breaks trace's sanity check.
            zero = torch.zeros_like(packed[:, 0:1, :])
            out_rows = 4 if self.dual_branch else 2
            return torch.cat([zero] * out_rows, dim=1)

        cond_y = packed[:, 0:1, :]
        artifact0 = packed[:, 1:2, :]
        batch_size = packed.shape[0]
        device = packed.device

        if self.training:
            t = torch.randint(0, self.num_steps, (batch_size,), device=device)
            u = torch.rand(batch_size, device=device)
            noise = torch.randn_like(artifact0)
        else:
            # Deterministic, reproducible validation step: spread fixed
            # timesteps across the schedule, use the interval midpoint
            # (u=0.5) for the continuous level, and a fixed (zero) epsilon so
            # the val loss is comparable across epochs.
            arange = torch.arange(batch_size, device=device, dtype=torch.long)
            t = (arange * (self.num_steps // max(batch_size, 1))) % self.num_steps
            u = torch.full((batch_size,), 0.5, device=device)
            noise = torch.zeros_like(artifact0)

        sqrt_abar, sqrt_one_minus = self.continuous_noise_level(t, u)
        # Single artifact class index 0 for the FACETpy gradient task.
        class_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Artifact branch.
        h_t_art = self.q_sample_continuous(artifact0, sqrt_abar, sqrt_one_minus, noise)
        pred_eps_art = self.predictor(h_t_art, cond_y, sqrt_abar, class_idx)

        if not self.dual_branch:
            return torch.cat([pred_eps_art, noise], dim=1)

        # Clean/EEG branch shares the SAME continuous level and epsilon (so the
        # two marginals are corrupted consistently for joint sampling).
        clean0 = packed[:, 2:3, :]
        h_t_clean = self.q_sample_continuous(clean0, sqrt_abar, sqrt_one_minus, noise)
        pred_eps_clean = self.predictor_clean(h_t_clean, cond_y, sqrt_abar, class_idx)

        return torch.cat([pred_eps_art, noise, pred_eps_clean, noise], dim=1)


class D4PMArtifactDataset:
    """Dataset packing per-channel diffusion training pairs.

    Single-branch (``dual_branch=False``): each example -> input ``(2, T)``
    with row 0 = noisy_y, row 1 = target artifact.

    Dual-branch (``dual_branch=True``): each example -> input ``(3, T)`` with
    row 0 = noisy_y, row 1 = artifact, row 2 = clean (read from
    ``clean_center``). This supervises BOTH marginals for the paper's joint
    posterior sampling.

    The diffusion target is ignored by the loss (epsilon-prediction self-target),
    so ``target`` is a ``(1, T)`` zeros placeholder. Channels are flattened so
    the checkpoint is channel-count independent.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dual_branch: bool = True,
        demean_input: bool = True,
        demean_target: bool = True,
        max_examples: int | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.dual_branch = bool(dual_branch)
        self.demean_input = bool(demean_input)
        self.demean_target = bool(demean_target)

        with np.load(self.path, allow_pickle=True) as bundle:
            self.noisy = bundle[_NOISY_KEY].astype(np.float32, copy=False)
            self.artifact = bundle[_ARTIFACT_KEY].astype(np.float32, copy=False)
            self.sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")
            if self.dual_branch:
                if _CLEAN_KEY not in bundle:
                    raise KeyError(
                        f"dual_branch=True requires '{_CLEAN_KEY}' in the NPZ bundle "
                        f"(keys: {list(bundle.keys())}). Use dual_branch=False to train the artifact branch only."
                    )
                self.clean = bundle[_CLEAN_KEY].astype(np.float32, copy=False)
            else:
                self.clean = None

        if self.noisy.ndim != 3 or self.artifact.ndim != 3:
            raise ValueError(
                f"D4PM dataset expects 3D arrays (examples, channels, samples), "
                f"got noisy={self.noisy.shape}, artifact={self.artifact.shape}"
            )
        if self.noisy.shape != self.artifact.shape:
            raise ValueError(
                f"noisy_center and artifact_center must match; got {self.noisy.shape} vs {self.artifact.shape}"
            )
        if self.clean is not None and self.clean.shape != self.noisy.shape:
            raise ValueError(
                f"clean_center must match noisy_center; got {self.clean.shape} vs {self.noisy.shape}"
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

        if self.dual_branch:
            clean_row = self.clean[base_idx, ch_idx : ch_idx + 1].astype(np.float32, copy=True)
            if self.demean_target:
                clean_row -= clean_row.mean(axis=-1, keepdims=True)
            packed = np.concatenate([noisy_row, artifact_row, clean_row], axis=0)
        else:
            packed = np.concatenate([noisy_row, artifact_row], axis=0)

        dummy_target = np.zeros((1, self.epoch_samples), dtype=np.float32)
        return packed, dummy_target

    @property
    def input_shape(self) -> tuple[int, int]:
        return (3 if self.dual_branch else 2, self.epoch_samples)

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
    """L1 (default) loss on packed (predicted epsilon, true epsilon) pairs.

    Handles both ``(B, 2, T)`` single-branch and ``(B, 4, T)`` dual-branch
    outputs, averaging the per-branch epsilon losses. L1 is the paper's Eq-1
    objective (critical to avoid over-smoothing).
    """

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
        n_rows = prediction.shape[1]
        if n_rows == 2:
            return self.loss_fn(prediction[:, 0:1, :], prediction[:, 1:2, :])
        if n_rows == 4:
            art = self.loss_fn(prediction[:, 0:1, :], prediction[:, 1:2, :])
            clean = self.loss_fn(prediction[:, 2:3, :], prediction[:, 3:4, :])
            return 0.5 * (art + clean)
        raise ValueError(f"D4PMEpsilonLoss expects (B, 2, T) or (B, 4, T) prediction, got {prediction.shape}")


def build_model(
    epoch_samples: int | None = 512,
    num_steps: int = 200,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    feats: int = 64,
    d_model: int = 128,
    d_ff: int = 512,
    n_heads: int = 2,
    n_layers: int = 3,
    embed_dim: int = 128,
    num_classes: int = 1,
    norm_first: bool = False,
    dual_branch: bool = True,
    lambda_snr: float = 1.0,
    input_shape: tuple[int, ...] | None = None,
    **_: object,
) -> D4PMTrainingModule:
    """CLI entry point for the paper-accurate model factory.

    Accepts (and ignores via ``**_``) the kwargs facet-train injects
    (``n_channels``, ``chunk_size``, ``sfreq``, ``target_type``,
    ``training_config``, ``target_shape``, ``context_epochs``, etc.).
    Explicit YAML ``model.kwargs`` override these defaults.
    """
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
        num_classes=num_classes,
        norm_first=norm_first,
        dual_branch=dual_branch,
        lambda_snr=lambda_snr,
    )


def build_loss(name: str = "l1", kind: str | None = None, **_: object) -> D4PMEpsilonLoss:
    """Loss factory. Accepts ``name`` (facet-train) or legacy ``kind``."""
    resolved = kind if kind is not None else name
    return D4PMEpsilonLoss(kind=resolved)


def build_dataset(
    path: str | None = None,
    context_path: str | None = None,
    max_examples: int | None = None,
    dual_branch: bool = True,
    demean_input: bool = True,
    demean_target: bool = True,
    **_: object,
) -> D4PMArtifactDataset:
    dataset_path = path or context_path
    if not dataset_path:
        raise ValueError("build_dataset requires path or context_path")
    return D4PMArtifactDataset(
        path=dataset_path,
        dual_branch=dual_branch,
        demean_input=demean_input,
        demean_target=demean_target,
        max_examples=max_examples,
    )
