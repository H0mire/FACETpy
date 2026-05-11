# D4PM Research Notes

## Source Papers

- **Primary**: Wang, Y. et al., 2025. "D4PM: A Dual-branch Driven Denoising
  Diffusion Probabilistic Model with Joint Posterior Diffusion Sampling for
  EEG Artifacts Removal." arXiv:2509.14302.
  Link: https://arxiv.org/abs/2509.14302
  Reference implementation: https://github.com/flysnow1024/D4PM
- **Background**: Ho, J., Jain, A., Abbeel, P., 2020. "Denoising Diffusion
  Probabilistic Models." NeurIPS 2020. arXiv:2006.11239.

## One-Paragraph Description

D4PM treats gradient-artifact removal as a generative source-separation
problem. Two independent denoising-diffusion probabilistic models (DDPMs) are
trained, each on a single signal class: an EEG branch that learns p(x_clean)
and an artifact branch that learns p(x_artifact). At inference time the two
reverse processes are run jointly. Each step predicts a candidate
x0_clean and a candidate x0_artifact; a data-consistency residual
r = y - (x0_clean + γ·x0_artifact) is computed against the noisy
observation y and split between the two branches via weights λ_dc·η and
λ_dc·(1−η)/γ. The result is "joint posterior sampling": both branches must
agree on a decomposition of the observation while still drawing from their
own learned manifolds.

## Key Architectural Components

The reference implementation (`denoising_model_eegdnet_class.py`) uses a
**transformer-style EEG-DNet noise predictor**, not the abstract page's claim
of a "U-Net". Each branch holds one independent copy of:

- `PositionalEncoding`: maps the scalar diffusion noise level (continuous
  α-bar value) to a sinusoidal embedding of dimension `feats`.
- `ClassEmbedding`: maps an artifact class index (0..C-1) to a learned
  embedding. We do not use class conditioning in our adaptation (single
  artifact class: gradient).
- Two parallel streams (`stream_x` for the noisy state x_t and `stream_cond`
  for the conditioning observation y):
  - Conv1d 1→feats, kernel 3, pad 1
  - Three EncoderLayers, each = (multi-head self-attention with
    d_model=512, d_k=d_v=64, n_heads=2, residual+LayerNorm) +
    (PoswiseFFN 512→2048→512, ReLU, residual+LayerNorm)
- Four FiLM modulation blocks fold (noise_level + class) embeddings into
  both streams.
- Output Conv1d feats→feats→1, kernel 3, pad 1 — predicts ε.

Reference config (`config/base.yaml`):

| Hyper | Reference | Our run |
|---|---|---|
| `feats` | 128 | 64 (smoke), 128 (full) |
| `d_model` | 512 | 256 (full) |
| `d_ff` | 2048 | 1024 (full) |
| `n_heads` | 2 | 2 |
| Encoder layers | 3 | 2 (full) |
| Diffusion steps T (training) | 500 | 200 (full) |
| Schedule | linear | linear |
| β range | 1e-4 → 0.02 | 1e-4 → 0.02 |
| Loss | L1(ε_pred, ε) | L1(ε_pred, ε) |
| Batch size | 1024 | 64 |
| Epochs | 4000 | 30 (full), 1 (smoke) |
| LR | 1e-3 | 1e-3 |
| Inference steps | T (full) | 50 (full eval), 10 (smoke) |

## Inputs Expected vs. Niazy Proof-Fit Mapping

The reference paper reports results on the **EEGdenoiseNet** benchmark:
single-channel 2-second segments at 256 Hz (512 samples) corrupted by EOG
or ECG artifacts. The adapted FACETpy task is gradient-artifact removal on
**Niazy proof-fit** segments.

| Aspect | Paper | Niazy proof-fit |
|---|---|---|
| Sampling rate | 256 Hz | 4096 Hz |
| Segment samples | 512 | 512 (resampled per epoch) |
| Channels | 1 (per-segment) | 30 EEG channels — we train per-channel |
| Artifact class | EOG / ECG | gradient (single class) |
| Pairs | 833 trigger-aligned 7-epoch contexts |

We use only `noisy_center` and `clean_center` and `artifact_center` from
the bundle (single epoch, no temporal context). Channel-wise inference
keeps the checkpoint independent of channel count.

## Loss Function

The original implementation uses **ε-prediction with L1 loss** between the
predicted noise and the true Gaussian noise ε added in the forward
diffusion step. We follow this exactly.

## Non-Obvious Training Tricks

- **Continuous-time noise levels**. Instead of training only on the discrete
  timesteps, the reference samples α-bar uniformly in the interval
  `[α_bar[t-1], α_bar[t]]` at each iteration. This gives the network a
  smoother conditioning signal.
- **L1 (not L2)** loss on ε is critical; L2 produced over-smooth artifact
  estimates in published ablations.
- **Two independently trained branches**. Each branch is trained on its own
  marginal distribution. Joint behavior emerges only at sampling time
  through the consistency constraint. Re-using one branch for both during
  joint sampling collapses the prior.

## Hardware Feasibility (RTX 5090, 24 GB VRAM)

A single branch matching the smoke config (2 encoder layers, d_model=256,
d_ff=1024, 512 samples per segment, 24990 single-channel examples = 833 ×
30) is comfortably small:

- Parameters per branch (smoke): roughly 2.5–3 M.
- Parameters per branch (full, d_model=256, layers=2): ~3–4 M.
- Memory per batch of 64 × 1 × 512 × float32: trivially under 1 GB.
- Wall-clock per epoch (smoke): expected < 30 s on a single RTX 5090.

We deviate from the reference config (feats=128, d_model=512, 3 layers,
T=1000, 4000 epochs) to fit our training budget. Trained for fewer epochs
on a smaller model. The ε-prediction loss is the same and convergence
behavior should be qualitatively similar; the absolute correlation
ceiling reported in the paper (CC > 0.99) is unlikely to be reached.

**Important caveat**: iterative sampling at inference time multiplies cost
by the number of reverse steps. With T=200 and 50 sampling steps, per-
channel inference of one 512-sample segment takes ~50 forward passes. For
the smoke run we use **10 sampling steps** so that the smoke job finishes
in well under one minute. For the full eval we use **50 sampling steps**.

## D4PM-Specific Adaptation Decisions

We implement a **single-branch conditional diffusion** form rather than the
two-branch joint posterior. Reasons:

1. The Niazy proof-fit dataset gives clean and artifact targets per epoch,
   so the artifact distribution is directly supervised.
2. Two independent diffusion branches double parameter count, training
   time, and inference cost. For a first FACETpy diffusion baseline, the
   one-branch artifact predictor still demonstrates the family
   characteristics (stochastic posterior sampling, ε-prediction L1 loss,
   iterative refinement).
3. The data-consistency residual reduces to a single subtraction at the
   correction stage when only the artifact branch is sampled:
   `clean_pred = y - artifact_pred`.

The conditioning input `y` is the noisy observation. The reverse process
samples `h_T ~ N(0, I)` and produces `h_0`, which is the gradient-artifact
estimate that is subtracted from `y`. The data-consistency soft constraint
is folded directly into each reverse step:

```
h0_pred  = predict_x0(h_t, ε_pred)
residual = y - h0_pred                        (since x_clean ≈ y - h)
h0_pred += λ_dc · residual                    (small λ_dc)
ε_pred   = (h_t - sqrt(α_bar_t) · h0_pred) / sqrt(1 - α_bar_t)
h_{t-1}  = posterior_mean + posterior_stddev · z
```

The dual-branch joint posterior remains as a future extension; we record
the reduction in `model_card.md`.

## Open Questions

1. The paper's "U-Net" claim contradicts the reference code's transformer
   architecture. We follow the code, not the abstract.
2. Native artifact epoch length varies (artifact_epoch_lengths_samples
   ranges across the dataset). The dataset has already been resampled to
   512 samples per epoch in the .npz bundle; we keep this convention.
3. The reference uses dynamic class conditioning (EOG vs ECG vs other);
   we drop class conditioning since gradient is a single class. This
   leaves the FiLM module degenerate but not removed, so future
   multi-artifact training can re-enable it without architecture changes.
4. The paper's λ_dc and η weights for joint posterior sampling are not
   tabulated; we set λ_dc=0.5 and rely on the single-branch reduction
   above.
