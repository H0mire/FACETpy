# D4PM paper-accuracy review

Source paper: **Shao et al., "A Dual-Branch Driven Denoising Diffusion
Probabilistic Model with Joint Posterior Diffusion Sampling for EEG Artifacts
Removal", arXiv:2509.14302** (5 pages, IEEE submission). Reference code:
https://github.com/flysnow1024/D4PM (`denoising_model_eegdnet_class.py`).
Background: Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models",
NeurIPS 2020 (arXiv:2006.11239).

This edition is the *paper-accurate* counterpart to the deliberately-reduced
`facet.models.d4pm`. Below: the discrepancy table (paper vs. original vs. this
edition), the EEG-fMRI applicability assessment, and the documented deviations.

## 1. Discrepancy table

| Aspect | Paper specifies | Original `d4pm` | This edition | Severity addressed |
|---|---|---|---|---|
| **Continuous noise-level conditioning** | `ε_θ` conditioned on `sqrt(ᾱ*_t) ~ U[sqrt(ᾱ_{t-1}), sqrt(ᾱ_t)]`, sampled per iteration (Sec 2.1); explicit fix for instability from discrete `t` | Discrete `sqrt(ᾱ_t)` at integer `t` for `q_sample` and embedding | Draw integer `t`, sample continuous `sqrt(ᾱ*)` in the interval; used for **both** `q_sample` and the embedding. Eval uses interval midpoint for reproducibility | **high** |
| **Transformer depth per path** | Three encoder blocks (Fig 2 "x3") | `n_layers=2` | `n_layers=3` default (configurable) | medium |
| **Block normalization** | Classic post-norm Add&Norm (reference EEG-DNet) | Post-norm (already matches) | Post-norm kept; `norm_first` flag added, default `False` | low |
| **Dual-FiLM + class label `z`** | Shared module embeds noise level **and** class `z` into `(γ, ξ)`; ablation Table 3 shows `z` lifts Base+Artifacts → full D4PM | FiLM on noise embedding only; no `z` | `ClassEmbedding` + `DualFiLM` fold `(noise_embed + class_embed)`; `num_classes=1` default | medium |
| **Dual-path shared FiLM + fusion** | Two identical paths share one Dual-FiLM; fused (added) then `(3x1, 1x1)` projection | Two streams added per layer; FiLM once post-fusion; output `Conv3→Conv3` | One shared `DualFiLM` per depth applied to **both** paths before fusion; output `Conv3 → ReLU → Conv1` | low |
| **Dual-branch joint posterior** | Two independent DDPMs (EEG + artifact) joined at inference (Algorithm 1) | Single artifact branch; consistency collapses to `h0 += λ_dc·(y − h0)` | Optional second clean branch (`dual_branch=True` default), supervised from `clean_center`; inference does the Algorithm-1 residual split | medium |
| **Mixture `y = x + x'·λ_SNR`** | Eq 2 with explicit SNR scaling | Implicit `λ_SNR=1` | Explicit `lambda_snr` (default `1.0`) in the joint-posterior residual | low |
| **Loss** | L1 on ε (Eq 1) | L1 default | L1 default; supports `(B,2,T)` and `(B,4,T)` | low |
| **Diffusion steps T** | Reference T=500 (up to 1000 typical) | `num_steps=200` | `num_steps=200` default (configurable; bump to 500 for full faithfulness) | low |
| **Inference sampler** | Exact DDPM posterior reverse update `μ_t = coef1·x̂0 + coef2·x_t`, `σ_t·η` for `t>1` (Alg 1 lines 18-21) | Ad-hoc DDIM-ish re-noising; posterior buffers unused | True ancestral update via `posterior_mean_coef1/coef2 + posterior_variance`, `η=0` on the final step; strided for speed | medium |

## 2. EEG-fMRI applicability assessment

| Paper method | Keep for EEG-fMRI? | Rationale |
|---|---|---|
| Continuous noise-level conditioning | **Yes** | Architecture- and data-agnostic, near-zero cost, the paper's core stabilization trick. Implemented faithfully. |
| L1 ε-prediction loss | **Yes** | Standard, cheap, critical for sharp artifact morphology vs over-smoothing. |
| Linear β schedule + exact DDPM posterior reverse update | **Yes** | Standard and cheap on CPU for 512-sample single-channel epochs. The posterior buffers existed before but were unused at inference; now used. |
| Three (x3) Transformer blocks per path | **Yes** | Faithful and still cheap at small `d_model`. Default 3; configurable down to 1 for the smoke. |
| Categorical artifact-class `z` + Dual-FiLM | **Yes (degenerate default)** | Gradient is a single class, so `z` is degenerate, but the ablation shows the pathway matters and future gradient+BCG training can use it. Wired with default class index 0. |
| Dual-branch (separate EEG + artifact DDPMs) + Joint Posterior Sampling | **Yes** | Feasible: the Niazy NPZ supplies both `clean_center` and `artifact_center`, so both marginals can be supervised. Default on; the smoke uses single-branch / tiny T. Largest fidelity gain over the original. |
| Mixture `y = x + x'·λ_SNR` with -5..5 dB SNR-sweep re-mixing | **No (re-mixing dropped)** | The synthetic SNR-sweep pairing is a dataset-construction trick for EEGdenoiseNet. The Niazy NPZ already contains real `noisy = clean + artifact` at the recorded gradient SNR, so we keep `λ_SNR=1` and do **not** re-mix. `lambda_snr` is exposed only for structural fidelity to the residual math. |
| Channel-wise / single-channel operation | **Yes** | The paper targets single-channel segments; FACETpy already trains and runs per channel, so the checkpoint is channel-count independent. |
| Per-epoch resample to model-domain samples then back | **Yes (FACETpy extension)** | Not in the paper; necessary for variable native gradient-epoch lengths at 4096 Hz. |

## 3. Documented deviations (intentional, FACETpy-appropriate)

1. **`num_classes=1` default** — single gradient artifact class; the class
   pathway acts as a learned global bias. Kept (not removed) so multi-artifact
   training is a config change with no architecture edit.
2. **`λ_SNR=1` default; no on-the-fly SNR mixing** — the Niazy proof-fit NPZ
   already encodes the recorded gradient mixture. The `lambda_snr` parameter
   exists only so the joint-posterior residual is structurally faithful to
   Eq 2 / Algorithm 1.
3. **Per-epoch resampling to `epoch_samples` then back to native length** — a
   FACETpy adaptation for variable-length gradient epochs; not in the paper.
4. **Optional single-branch mode and strided ancestral sampling** — for cheap
   CPU runs; the faithful defaults are dual-branch with the full ancestral
   schedule.
5. **`num_steps=200` default** — a budget compromise vs the reference T=500;
   configurable.
6. **Eval-step determinism** — during validation the module uses the interval
   midpoint (`u=0.5`) and zero ε so the validation loss is reproducible across
   epochs; training uses the full random continuous sampling.

## 4. Note on author attribution

The arXiv identifier **2509.14302** is the authoritative anchor for this work;
the FACETpy `research_notes.md` for the original model lists the first author as
"Wang, Y. et al." while this task's analysis attributes it to "Shao et al." The
method, equations, and reference repository are identical regardless; the
implementation here follows the reference code
(`denoising_model_eegdnet_class.py`) and the paper's Algorithm 1.
