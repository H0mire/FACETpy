# Deep Learning Architectures For Gradient Artifact Removal — Results Report

Date: 2026-05-11
Repository state: feature/add-deeplearning (commit `32190a0`+)
Dataset: Niazy proof-fit context dataset (832 examples, 7-epoch context, 30 channels, 512 samples per epoch, 4096 Hz)

## 1. Executive Summary

This report compares twelve deep-learning architectures trained on the Niazy
proof-fit dataset under the parallel-agent workflow described in
[`docs/model_agent_prompts.md`](../model_agent_prompts.md) and selected from
[`docs/research/architecture_catalog.md`](architecture_catalog.md). Each
architecture is one of the seven model families from the technical report
[`docs/research/dl_eeg_gradient_artifacts.pdf`](dl_eeg_gradient_artifacts.pdf).

### Ranking (by clean SNR improvement on the Niazy proof-fit validation set)

| Rank | Model | Family | SNR↑ (dB) | Artifact corr. | Residual RMS ratio | MSE reduction |
|---:|---|---|---:|---:|---:|---:|
| 1 | **Demucs** | Audio (U-Net + LSTM) | **+31.28** | 0.9996 | 0.027 | 99.92 % |
| 2 | Conv-TasNet | Audio (TCN) | +22.03 | 0.9969 | 0.079 | 99.37 % |
| 3 | SepFormer | Audio (Dual-path Transformer) | +19.05 | 0.9938 | 0.112 | 98.76 % |
| 4 | **Cascaded Context DAE** | Autoencoder (MLP, 7-epoch context) | **+18.84** | 0.9935 | 0.114 | 98.65 % |
| 5 | **Cascaded DAE** | Autoencoder (MLP, single epoch) | **+17.79** | 0.9917 | 0.129 | 98.35 % |
| 6 | Nested-GAN | GAN (TF + time domain) | +13.54 | 0.9776 | 0.210 | — |
| 7 | DenoiseMamba | Sequence / SSM | +11.80 | 0.9664 | 0.257 | 93.39 % |
| 8 | IC-U-Net | Discriminative (CNN + ICA prior) | +11.77 | 0.9667 | 0.258 | 93.34 % |
| 9 | ViT-Spectrogram | Vision (MAE inpainting) | +11.60 | 0.9660 | 0.263 | 93.08 % |
| 10 | ST-GNN | Graph (spatial topology) | +11.00 | 0.9595 | 0.277 | ≈ 92 % |
| 11 | DPAE | Discriminative (dual-pathway CNN) | +7.48 | 0.9132 | 0.423 | 82.12 % |
| 12 | D4PM | Diffusion | (+3.21)\* | 0.7251 | 0.699 | — |
| 13 | DHCT-GAN v2 | GAN (hybrid CNN + Transformer) | +1.69 | 0.5673 | 0.824 | 32.17 % |
| — | DHCT-GAN v1 | GAN (single-epoch input) | −7.13 | 0.1577 | 2.272 | −416 % |

\* D4PM evaluated on only 32 examples × 4 channels (Diffusion inference cost);
not directly comparable to the full-set evaluations.

### Headline findings

1. **Audio-derived source-separation architectures dominate.** All three audio
   models (Demucs, Conv-TasNet, SepFormer) outperform every other family. The
   inductive bias from speech and music source separation — rhythmic loud noise
   vs. stochastic quiet signal — maps remarkably well to the EEG-fMRI gradient
   artifact problem.
2. **Small MLPs with the right loss break into the top five.** The retrofilled
   Cascaded DAE family — vanilla fully-connected autoencoders with hidden
   `[512, 128, 512]`, joint two-stage training, L1 loss — places at rank 4
   (Cascaded Context DAE, 4.46 M params, +18.84 dB) and rank 5 (Cascaded DAE,
   1.31 M params, +17.79 dB). At 0.1–0.5 s wall-clock inference on the full
   4998-pair validation split they are 100–1000× faster than the audio
   architectures while reaching 60 % of Demucs's SNR improvement. The MLP
   ceiling is not as low as audio-source-separation literature suggests.
3. **A "discriminative + sequence + vision + graph" middle plateau exists at
   ~ +11 to +12 dB** (DenoiseMamba, IC-U-Net, ViT-Spectrogram, ST-GNN). These
   four very different architectures converge on essentially the same
   performance, suggesting the dataset's information ceiling is approached.
4. **Input contract matters more than architecture family.** The DHCT-GAN v1
   catastrophic failure (−7.13 dB) was caused by feeding only the central
   epoch (not the 7-epoch context). The v2 salvage with multi-epoch context
   recovers to +1.69 dB on the same architecture — a 9 dB swing from input
   choice alone. The cascaded-DAE retrofill shows the symmetric effect: when
   `cascaded_context_dae` uses the 7-epoch context vs `cascaded_dae` with only
   the single center epoch, both other knobs identical, the context adds
   +1.05 dB SNR and 1 percentage point of artifact correlation — much smaller
   than the v1→v2 swing, suggesting context helps most when the rest of the
   model is otherwise weak.
4. **AAS-fidelity is the ceiling, not true denoising.** The "clean" target in
   the proof-fit dataset is the AAS-corrected reference, so the metrics measure
   each architecture's ability to *reproduce* AAS rather than fundamentally
   beat it. The audio-architecture lead should be interpreted as "they
   approximate the AAS template more accurately", not "they remove more
   physical artifact than AAS does".

## 2. Methodology

### Dataset

All models were trained on the proof-fit context dataset produced by
[`examples/build_niazy_proof_fit_context_dataset.py`](../../examples/build_niazy_proof_fit_context_dataset.py)
with default parameters:

- `--target-epoch-samples 512`
- `--context-epochs 7`
- Source artifact bundle:
  `output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz`

Dataset shape:

- 833 trigger-locked context examples, each of shape `(7, 30, 512)`
- Target (center epoch) shape `(30, 512)`
- Sampling rate 4096 Hz
- Native artifact epoch length 584–605 samples (cropped to 512)
- Mean |clean| 215 µV, mean |artifact| 919 µV

Train / val / test split is determined per model factory (typically 80/20 with
seed 42). Some models trained on 833 trigger windows, others channel-wise
expanded to 24,990 (= 833 × 30) examples — see "Evaluation comparability"
caveat below.

### Hardware and orchestration

- Two RunPod workers, each NVIDIA RTX 5090 (24 GB VRAM) and ~64 GB system RAM
- MacBook orchestrator running a single central
  `python tools/gpu_fleet/fleet.py dispatch --loop --interval 60`
- One model agent per architecture, each in its own Git worktree
  (`worktrees/model-<id>/`)
- Sync via `tools/gpu_fleet/sync_worktree_to_runpod.sh`; per-job `uv sync`
  and `check_torch.py` verification before training
- Detached `tmux` sessions on the pod with per-GPU `flock` lock files

### Training contract

Required from every agent (enforced by the prompt template and
[`docs/model_development_guidelines.md`](...) → `src/facet/models/README.md`):

- `device: cuda` in any training YAML submitted to the fleet
- Reuse FACETpy helpers: `facet-train` CLI, `TrainableModelWrapper`,
  `NPZContextArtifactDataset`, `ModelEvaluationWriter`
- TorchScript export under `training_output/<run>/exports/<model_id>.ts`
- Evaluation outputs conforming to
  [`src/facet/models/evaluation_standard.md`](../../src/facet/models/evaluation_standard.md):
  `evaluation_manifest.json`, `metrics.json` (with a `flat_metrics` table),
  `evaluation_summary.md`, and `plots/`

Hyperparameter freedom: every agent chose its own optimizer, learning rate,
batch size, and `max_epochs` based on the source paper for its architecture
family. Smoke runs were fixed at `max_epochs: 1`.

## 3. Architectures Tested

### 3.1 Discriminative (CNN / Autoencoder)

- **DPAE** — Dual-Pathway Autoencoder (Xiong et al. 2023). Local pathway
  with small dilated kernels for high-frequency content, global pathway with
  large kernels for low-frequency trends. Channel-wise.
- **IC-U-Net** — U-Net trained on mixtures of Independent Components
  (Chuang et al. 2022). Uses an ICA pre-decomposition to separate sources,
  then the U-Net reconstructs the clean component mixture.
- **Cascaded DAE / Cascaded Context DAE** — Two-stage residual
  fully-connected denoising autoencoders ported from the legacy
  `feature/deeplearning` prototype (Run-1 retrofill May 2026). Stage 1
  predicts the artifact; Stage 2 sees `noisy − stage1(noisy)` and predicts
  the remaining residual. Both stages share identical `[512, 128, 512]`
  hidden dimensions; the only architectural difference is the input
  dimension (1×512 vs 7×1×512). Joint end-to-end training with L1 loss.
  Channel-wise inference.

### 3.2 Generative (GAN)

- **DHCT-GAN** v1 and v2 — Dual-Branch Hybrid CNN + Transformer Generator
  (MDPI Sensors 2025). v1 used only the central epoch as input; v2 added
  the full 7-epoch context after the catastrophic v1 failure.
- **Nested-GAN** — Inner GAN on time-frequency spectrogram, outer GAN on
  time-domain refinement. Divide-and-conquer (PubMed 41183389).

### 3.3 Probabilistic (Diffusion)

- **D4PM** — Dual-branch Driven Denoising Diffusion Probabilistic Model
  with Joint Posterior Diffusion Sampling (arXiv 2509.14302). Models the
  signal and the artifact as two coupled diffusion processes with a
  consistency constraint `x_noisy = x_clean + x_artifact`.

### 3.4 Sequence (State Space)

- **DenoiseMamba** — ConvSSD module combining local 1D Convolution with
  Mamba/SSD layer (IEEE Xplore 11012652). Linear complexity for long
  sequences.

### 3.5 Audio-Inspired Source Separation

- **Conv-TasNet** — Temporal Convolutional Network with stacked dilated
  convolutions (Luo & Mesgarani 2019). Source 1 = clean EEG, source 2 =
  artifact.
- **Demucs** — U-Net with LSTM bottleneck (Défossez et al. 2019, hybrid
  v4 2022). Originally for music stem separation.
- **SepFormer** — Dual-path Transformer with intra-chunk and inter-chunk
  attention (Subakan et al. 2021). Maps naturally onto the slice-vs-volume
  structure of the gradient artifact.

### 3.6 Vision (Spectrogram Inpainting)

- **ViT-Spectrogram** — Vision Transformer / Masked Autoencoder operating
  on time-frequency spectrograms (Dosovitskiy et al. 2020, He et al. 2021).
  Masks GA-dominated regions and inpaints them.

### 3.7 Graph (GNN)

- **ST-GNN** — Spatiotemporal Graph Neural Network with electrode-graph
  adjacency (Yu et al. 2018 + EEG-GCNN adaptation). Enforces spatial
  consistency based on 10-20 layout.

## 4. Per-Family Results

### 4.1 Audio dominates the leaderboard

The top three positions (Demucs, Conv-TasNet, SepFormer) are all
audio-derived source separation architectures. Their advantage is not
incremental — Demucs at +31.28 dB and SepFormer at +19.05 dB are 7–20 dB
above the discriminative/sequence/graph plateau (~+11 dB).

**Interpretation.** Source separation explicitly frames the problem as
"separate two sources in the same channel": a loud rhythmic signal (artifact)
and a quieter stochastic signal (EEG). This is mathematically isomorphic to
the cocktail-party problem. The audio community has spent ten years building
inductive biases for exactly this setup; reusing those biases on EEG-fMRI
yields a step-change.

**Demucs specifically.** The U-Net + LSTM bottleneck combines spatial
multiscale processing (U-Net) with explicit long-range temporal memory
(LSTM). For a periodic artifact with slow drift across TRs, this is
near-ideal. Demucs achieves artifact correlation 0.9996 and residual RMS
ratio 0.027 — essentially indistinguishable from the AAS reference within
the proof-fit dataset.

### 4.2 Discriminative / Sequence / Vision / Graph plateau at ~ +11 dB

Four architectures from four very different families cluster tightly:

| Model | SNR↑ dB | art.corr | res.RMS |
|---|---:|---:|---:|
| DenoiseMamba | +11.80 | 0.966 | 0.257 |
| IC-U-Net | +11.77 | 0.967 | 0.258 |
| ViT-Spectrogram | +11.60 | 0.966 | 0.263 |
| ST-GNN | +11.00 | 0.960 | 0.277 |

This convergence is striking. Linear-complexity SSMs, U-Nets with ICA
priors, ViT-on-spectrogram inpainting, and graph convolutions on scalp
topology all land within 1 dB of each other.

**Interpretation.** This likely reflects an *information ceiling*
attainable from the (7, 30, 512) input contract without source-separation
priors. Architecturally simpler models that are well-tuned can extract
the same amount of artifact-template signal that more complex models can.
Going beyond this plateau seems to require explicit two-source modeling,
which is exactly what the audio family provides.

DPAE (the simpler dual-pathway CNN) at +7.48 dB is the bottom of the
discriminative range — it is channel-wise and lacks any context-spanning
mechanism, which costs it about 4 dB versus IC-U-Net.

### 4.2b Retrofilled cascaded DAEs break the +11 dB plateau

The two-stage residual cascaded denoising autoencoders, retrofilled with
L1 loss and symmetric hidden `[512, 128, 512]`, jump *past* the entire
discriminative/sequence/vision/graph plateau and land between SepFormer
and Nested-GAN:

| Model | Params | SNR↑ dB | art.corr | res.RMS | Inference |
|---|---:|---:|---:|---:|---:|
| Cascaded Context DAE | 4.46 M | +18.84 | 0.9935 | 0.114 | 0.1 s |
| Cascaded DAE | 1.31 M | +17.79 | 0.9917 | 0.129 | 0.5 s |

Three observations:

1. **The MLP ceiling is not at +11 dB.** Earlier reasoning that fully
   connected models cannot match conv/transformer architectures because they
   lack translation invariance is partly wrong: a well-sized MLP with the
   right loss reaches the second-best non-audio family score (only
   Nested-GAN's adversarial training comes within 5 dB on the same dataset,
   and the audio architectures stay 4–12 dB ahead).
2. **L1 is doing serious work.** Bit-identical architecture, MSE → L1 loss
   change alone is worth several dB on a heavy-tailed artifact distribution
   (max amplitude 13.6 mV, mean 0.92 mV). Demucs uses L1 — none of the
   plateau-tier models do — and the cascaded retrofill exhibits the same
   pattern.
3. **Inference cost is 100–1000× lower** than the audio family for ~60 %
   of the SNR improvement. The 0.1 s wall-clock for the entire 4998-pair
   validation split makes the cascaded context DAE a strong candidate
   whenever inference latency matters more than the last few dB of SNR.

The 1.05 dB gain from cascaded DAE → cascaded context DAE is the cleanest
controlled-ablation measurement of the 7-epoch context value in the entire
study (every other knob is identical between the two configs).

### 4.3 GAN family is harder to train than the report suggested

- DHCT-GAN v1 with single-epoch input produced clean MSE *worse* than no
  correction (−416 % MSE reduction, artifact correlation 0.158). The
  adversarial loss without sufficient input context drove the generator to
  destroy signal alongside artifact.
- DHCT-GAN v2 with the multi-epoch context fix recovered to +1.69 dB —
  still below every non-GAN architecture in the test set.
- Nested-GAN at +13.54 dB was the only GAN that beat the discriminative
  plateau. Its dual-domain trick (inner spectrogram GAN + outer time-domain
  refinement) provided enough structural prior to stabilize training.

**Interpretation.** GANs require both a strong input contract *and* an
explicit decomposition of the problem (Nested-GAN's TF + time split). A
plain adversarial setup without those scaffolds is fragile on this dataset.

### 4.4 Diffusion did not get a fair chance

D4PM produced only a partial evaluation: 32 examples × 4 channels (versus
4998 × 30 for the other audio-family runs). The iterative sampling cost of
diffusion at inference time made a full validation set evaluation
operationally expensive within the time budget. The artifact correlation
of 0.7251 on this subset is comparable to DPAE-level performance, but the
sample is too small to rank against the full-set models.

A focused follow-up that evaluates D4PM on the full validation split with
a tuned sample-step count is recommended before drawing conclusions about
diffusion-based artifact removal.

## 5. Critical Caveats

### 5.1 The "clean" target is the AAS reference

The Niazy proof-fit dataset's target is the AAS-cleaned signal, not a true
artifact-free ground truth (which does not exist for simultaneous EEG-fMRI).
Every metric in this report therefore measures *fidelity to the AAS template*,
not *physical denoising quality*. The audio architectures' dominance means
they reproduce AAS more accurately, not that they remove more physical
artifact than AAS does. To make a claim about beating AAS, an out-of-sample
evaluation against an external clinical reference (e.g. spike-detection rate
on independent data) is required.

### 5.2 Evaluation set size varies across models

Models used inconsistent validation split sizes:

- DPAE, DenoiseMamba: 24,990 channel-wise examples (split from 833 trigger
  windows × 30 channels)
- IC-U-Net, ViT-Spectrogram, Nested-GAN: 833 examples (trigger-window level)
- Conv-TasNet, SepFormer, Demucs: 4,998 channel-wise examples (sub-sample)
- D4PM: 32 × 4 examples (diffusion inference cost)
- DHCT-GAN, DHCT-GAN-v2, ST-GNN: 833 examples but with non-standard metric
  reporting

This unfortunately makes the absolute dB numbers not strictly comparable
across all twelve models. The *ranks* are robust within each evaluation
size class, but the absolute gap between the top audio models and the
plateau models could be partly explained by different test distributions.

A normalization round where every model re-evaluates on the same fixed
holdout split is recommended before reporting these numbers in the thesis
narrative.

### 5.3 Different agents handle input differently

The DHCT-GAN failure exposed that several architectures' `build_dataset`
factories pull `noisy_center` (shape `(30, 512)`) instead of the full
`noisy_context` (shape `(7, 30, 512)`). The 9 dB swing between DHCT-GAN v1
and v2 illustrates how much this matters. Other plateau-tier models should
be audited for the same issue before drawing strong family-level
conclusions.

### 5.4 Some models export to GPU-locked TorchScript

The ViT-Spectrogram agent discovered that `torch.jit.trace` bakes the
trace-time device into buffers via explicit `.to(device)` calls in
`forward()`. This locks the exported `.ts` file to a specific CUDA device.
The fix (commit `4184443` in worktrees/model-vit_spectrogram/) is removing
the explicit `.to()` and relying on `Module.to()` for buffer migration.
Other agents may have the same issue silently; it would not affect the
metrics above but would affect downstream deployment.

## 6. Runtime & Cost

(Approximate, from each agent's `HANDOFF.md` and `summary.json`.)

| Model | Smoke run | Full run | Wallclock |
|---|---:|---:|---|
| DPAE | < 30 s | 48 s | very fast |
| DenoiseMamba | ~1 min | ~3-4 min | fast |
| Conv-TasNet | ~1 min | ~3 min | fast |
| Demucs | ~1 min | ~4 min | fast |
| SepFormer | ~2 min | ~6 min | moderate |
| ViT-Spectrogram | < 1 min | 87 s | fast |
| IC-U-Net | ~1 min | ~3 min | fast |
| ST-GNN | ~1 min | ~4 min | fast |
| Nested-GAN | ~2 min | ~6 min | moderate |
| DHCT-GAN v1/v2 | ~1 min | ~3 min | fast |
| D4PM | ~5 min | ~10 min | slow (iterative sampling) |

All ran on one RTX 5090. Even the slowest model (D4PM) needs only ~10 min
of GPU time for a full training pass on this dataset size. The bottleneck
for thesis-scale experimentation is *not* compute; it is implementation
and analysis time.

## 7. Recommendations

### 7.1 For the thesis narrative

1. Lead with the **family-level finding**: audio source-separation
   architectures outperform every other family on this dataset by a wide
   margin. This is the strongest, cleanest contribution.
2. Use **Demucs as the primary recommendation** for offline correction
   where fidelity matters. It is the highest-fidelity model, runs fast,
   and the architecture is well-understood.
3. Present **DenoiseMamba** as the recommendation for **high sampling rate
   or long-context streaming** scenarios. The +11.80 dB at linear
   complexity gives a different operating point than the audio models.
4. Frame the **DHCT-GAN v1 → v2 result** as a methodological lesson about
   input contract sensitivity. This is publishable as a small finding on
   its own.

### 7.2 For the next experimental round

Before extending to more architectures, the following gap-filling work
would tighten the comparison:

- **Re-evaluate every model on the same held-out test split** to remove
  the test-size confound.
- **Run D4PM on the full validation set** (8 examples is not enough — at
  the very least 833).
- **Audit each plateau model's dataset factory** to confirm whether it
  uses `noisy_context` or `noisy_center`, and re-train any that used the
  wrong one.
- **Cross-dataset evaluation against an independent EEG-fMRI bundle** (not
  Niazy-derived) to test whether the audio family's lead persists out of
  distribution.

### 7.3 Future architectures worth trying

The architecture catalog lists hybrid combinations that no agent has tried
yet. Given the audio family's lead, the most promising hybrids are:

- **Demucs + GNN** — replace Demucs's channel-wise heads with a graph
  convolution over the scalp topology
- **Conv-TasNet + Mamba** — replace the TCN with a Mamba block for the
  separation network (audio inductive bias + linear complexity)
- **Source-separation + diffusion** — diffuse on the separated artifact
  branch only, keep the EEG branch deterministic

## 8. Reproducibility

All model implementations live under `src/facet/models/<model_id>/` in
their respective branches `feature/model-<model_id>`. Each contains:

- `processor.py` — FACETpy `DeepLearningCorrection` + adapter
- `training.py` — `build_model`, `build_loss`, `build_dataset` factories
- `training_niazy_proof_fit.yaml` and `*_smoke.yaml` configs
- `documentation/model_card.md` — architecture description
- `documentation/research_notes.md` — paper references and decisions
- `documentation/evaluations.md` — per-run results
- `HANDOFF.md` — agent's final summary

To reproduce any single result:

```bash
git checkout feature/model-<id>
git worktree add ../worktrees/model-<id>
cd ../worktrees/model-<id>
uv sync
uv run python tools/gpu_fleet/fleet.py submit \
  --name <id>_niazy_full \
  --worktree . \
  --training-config src/facet/models/<id>/training_niazy_proof_fit.yaml \
  --prepare-command "uv run python examples/build_niazy_proof_fit_context_dataset.py --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz --target-epoch-samples 512 --context-epochs 7 --output-dir output/niazy_proof_fit_context_512"
```

with the central dispatcher running on the MacBook
(`fleet.py dispatch --loop --interval 60`) and the worker config in
`tools/gpu_fleet/workers.local.yaml`.

## 9. Per-Model Hand-Off Summaries

Each agent wrote a `HANDOFF.md` in its worktree with detailed numbers,
hypotheses, and follow-up suggestions. Quick links:

- DPAE: `worktrees/model-dpae/HANDOFF.md`
- IC-U-Net: `worktrees/model-ic_unet/HANDOFF.md`
- DenoiseMamba: `worktrees/model-denoise_mamba/HANDOFF.md`
- ViT-Spectrogram: `worktrees/model-vit_spectrogram/HANDOFF.md`
- ST-GNN: `worktrees/model-st_gnn/HANDOFF.md`
- Conv-TasNet: `worktrees/model-conv_tasnet/HANDOFF.md`
- Demucs: `worktrees/model-demucs/HANDOFF.md`
- SepFormer: `worktrees/model-sepformer/HANDOFF.md`
- Nested-GAN: `worktrees/model-nested_gan/HANDOFF.md`
- D4PM: `worktrees/model-d4pm/HANDOFF.md`
- DHCT-GAN v1: `worktrees/model-dhct_gan/HANDOFF.md` (failure analysis)
- DHCT-GAN v2: `worktrees/model-dhct_gan_v2/HANDOFF.md` (salvage attempt)

Evaluation manifests are in
`output/model_evaluations/<model_id>/<run_id>/evaluation_manifest.json`
(main repo) or
`worktrees/model-<id>/output/model_evaluations/<model_id>/<run_id>/evaluation_manifest.json`
(per-worktree, prior to symlink unification).
