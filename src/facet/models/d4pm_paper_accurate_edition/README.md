# D4PM (paper-accurate edition)

A more faithful re-implementation of **D4PM** for FACETpy gradient-artifact
removal, kept as a *separate* model package from the original
`facet.models.d4pm` (which is a deliberate single-branch reduction).

> Shao et al., "A Dual-Branch Driven Denoising Diffusion Probabilistic Model
> with Joint Posterior Diffusion Sampling for EEG Artifacts Removal",
> arXiv:2509.14302. Reference code:
> https://github.com/flysnow1024/D4PM (`denoising_model_eegdnet_class.py`).

The original was intentionally simplified; this edition restores the paper's
defining components while staying compatible with the facet-train factory
contract and runnable on CPU.

## What changed vs. the original `d4pm`

| # | Paper component | Original `d4pm` | This edition |
|---|---|---|---|
| 1 | **Continuous noise-level conditioning** `sqrt(ᾱ*_t) ~ U[sqrt(ᾱ_{t-1}), sqrt(ᾱ_t)]` (Sec 2.1) | Discrete `sqrt(ᾱ_t)` at integer `t` | Draws integer `t`, then samples the continuous level and uses it for **both** `q_sample` and the conditioning embedding (deterministic interval midpoint in eval) |
| 2 | **Three Transformer blocks per path** (Fig 2 "x3") | `n_layers=2` default | `n_layers=3` default (configurable) |
| 3 | **Dual-FiLM with class label `z`** (Sec 2.2) | FiLM on noise embedding only, no `z` | `ClassEmbedding` + `DualFiLM` fold `(noise_embed + class_embed)`; default `num_classes=1` → learned global bias for the single gradient class |
| 4 | **Shared Dual-FiLM inside each path; `(3x1, 1x1)` output head** | FiLM applied once post-fusion; output `Conv3→Conv3` | One shared `DualFiLM` per depth applied symmetrically to **both** paths before fusion; output `Conv3 → ReLU → Conv1` |
| 5 | **Dual-branch + Joint Posterior Sampling** (Algorithm 1) | Single artifact branch only | Optional second clean/EEG branch (`dual_branch=True` default) supervised from `clean_center`; inference does the Algorithm-1 residual split |
| 6 | **Mixture `y = x + x'·λ_SNR`** (Eq 2) | Implicit `λ_SNR=1` | Explicit `lambda_snr` parameter (default `1.0`) in the joint-posterior residual |
| 7 | **Exact DDPM ancestral sampler** (Alg 1 lines 18-21) | Ad-hoc re-noising (`sqrt(ᾱ_prev)·h0 + sqrt(1-ᾱ_prev)·noise`), posterior buffers unused | True ancestral update via `posterior_mean_coef1/coef2 + posterior_variance`, `η=0` on the final step |
| 8 | **Post-norm Transformer blocks** | Post-norm | Post-norm kept (faithful to EEG-DNet); `norm_first` flag exposed, defaults `False` |
| 9 | **L1 ε-prediction loss** (Eq 1) | L1 default | L1 default kept; handles both `(B,2,T)` and `(B,4,T)` outputs |

## Architecture (per branch)

```
h_t ─ Conv1d(1→feats, k3) ─ Linear(feats→d_model) ─┐
                                                    ├─ [TransformerEncoderLayer1D × n_layers]
y   ─ Conv1d(1→feats, k3) ─ Linear(feats→d_model) ─┘   (post-norm self-attn + FFN)
        shared DualFiLM(noise_embed + class_embed) applied to BOTH paths each layer
        paths fused by addition → Linear(d_model→feats) → Conv1d(feats→feats, k3) → ReLU → Conv1d(feats→1, k1) → ε
```

`dual_branch=True` instantiates two such branches with **independent** weights
(no sharing, per Fig 2): `predictor` (artifact) and `predictor_clean` (EEG).

## Training (facet-train factory contract)

`training.py` exposes:

- `build_model(**kwargs)` → `D4PMTrainingModule`. Accepts the injected
  facet-train kwargs via `**_`; explicit YAML `model.kwargs` override defaults.
- `build_loss(name="l1", ...)` → `D4PMEpsilonLoss` (also accepts legacy `kind`).
- `build_dataset(path=..., dual_branch=True, ...)` → `D4PMArtifactDataset`.

The dataset packs per channel-example into the model input:

- single-branch → `(2, T)` = `[noisy_y, artifact]`
- dual-branch → `(3, T)` = `[noisy_y, artifact, clean]`

The diffusion target is the self-sampled ε, so the dataset `target` is a
`(1, T)` zeros placeholder ignored by the loss. The module samples the
continuous level and ε internally and returns the packed
`(pred_ε, true_ε)` pairs.

See `training_niazy_proof_fit_smoke.yaml` for an illustrative tiny-dims CPU
config (not executed by the test suite).

## Inference

`processor.py` registers `D4PMPaperAccurateCorrection` under the
**globally-unique** name `d4pm_paper_accurate_correction` (the original uses
`d4pm_correction`). The adapter resamples each native trigger epoch to
`epoch_samples`, runs the reverse sampler per channel, and resamples the
predicted artifact back to the native length (a FACETpy-specific extension for
variable-length gradient epochs at 4096 Hz).

- `dual_branch=True`: Joint Posterior Sampling. Per reverse step, predict
  `x0` (clean) and `x'0` (artifact), compute `r = y − (x0 + x'0·λ_SNR)`, apply
  `x̂0 = x0 + λ_dc·r` and `x̂'0 = x'0 + (1−λ_dc)·r`, then run the true DDPM
  ancestral update for both branches. Returns the artifact estimate.
- `dual_branch=False`: single artifact branch with the documented
  `x0 += λ_dc·(y − x0)` data-consistency reduction.

## Documented deviations (FACETpy-appropriate)

- **`num_classes=1` default** — gradient is a single artifact class, so `z` is
  degenerate (learned global bias). Wired so multi-artifact (gradient + BCG)
  training is a config change.
- **`λ_SNR=1` default** — the Niazy proof-fit NPZ already encodes
  `noisy = clean + artifact` at the recorded SNR, so the synthetic SNR-sweep
  re-mixing (-5..5 dB) from EEGdenoiseNet is dropped as inapplicable; the
  parameter exists only for structural fidelity to Eq 2 / Algorithm 1.
- **Per-epoch resample to `epoch_samples` then back** — not in the paper; a
  FACETpy extension for variable native gradient-epoch lengths.
- **Optional single-branch / strided sampling** — for cheap CPU runs.

See `documentation/paper_accuracy_review.md` for the full discrepancy table and
EEG-fMRI applicability assessment.
