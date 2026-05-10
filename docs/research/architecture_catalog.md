# Architecture Catalog For Model Agents

This is the menu of deep-learning architectures that FACETpy model agents can
implement and evaluate against the Niazy proof-fit dataset. It is derived from
[`dl_eeg_gradient_artifacts.pdf`](dl_eeg_gradient_artifacts.pdf) (Comprehensive
Technical Report on Deep Learning Architectures for Gradient Artifact
Reconstruction and Removal in Simultaneous EEG-fMRI).

The orchestrator picks one architecture per spawned agent. The agent then
researches the cited paper, designs the implementation, and trains. Multiple
agents implementing different architectures run in parallel through the GPU
fleet queue.

## How To Use

1. The orchestrator picks an entry from the catalog.
2. The orchestrator spawns a model agent with the prompt template from
   `docs/model_agent_prompts.md`, filling in the chosen architecture.
3. The agent reads the report PDF for the family overview, then performs
   independent research on the specific paper before implementing.
4. The agent works under `.claude/worktrees/model-<model_id>/` and submits to
   the GPU fleet via `tools/gpu_fleet/fleet.py submit`.
5. The orchestrator runs `fleet.py dispatch --loop` centrally on the MacBook.

## Architecture Families

### 1. Discriminative (CNN / Autoencoder)

Direct supervised regression: noisy input → clean (or artifact) target.

| Model id (suggested) | Source model in report | Notes |
|---|---|---|
| `dpae` | DPAE — Dual-Pathway Autoencoder | Local + global pathway, high vs low frequency separation. |
| `ic_unet` | IC-U-Net | U-Net trained on mixtures of Independent Components. |

### 2. Generative (GAN)

Reconstruction loss + adversarial discriminator. Targets high-frequency
fidelity instead of MSE blur.

| Model id (suggested) | Source model in report | Notes |
|---|---|---|
| `dhct_gan` | DHCT-GAN | Dual-branch CNN + Transformer generator, splits clean and artifact features. |
| `nested_gan` | Nested-GAN | Inner GAN on time-frequency, outer GAN on time domain. |

### 3. Probabilistic (Diffusion)

Iterative denoising; treats artifact removal as joint posterior source
separation.

| Model id (suggested) | Source model in report | Notes |
|---|---|---|
| `d4pm` | D4PM — Dual-branch Driven Denoising Diffusion | Joint posterior sampling with consistency constraint x_clean + x_artifact = x_noisy. |

### 4. Sequence Modeling (State Space / Mamba)

Linear-complexity sequence models, designed to handle long EEG epochs at high
sampling rate.

| Model id (suggested) | Source model in report | Notes |
|---|---|---|
| `denoise_mamba` | DenoiseMamba | ConvSSD module, 1D conv + Mamba/SSD layer for long-range periodicity. |

### 5. Audio-Inspired Source Separation

Treats artifact removal as a cocktail-party-like blind source separation.

| Model id (suggested) | Source model in report | Notes |
|---|---|---|
| `conv_tasnet` | Conv-TasNet | Time-domain TCN with stacked dilated convolutions. |
| `demucs` | Demucs | U-Net + LSTM, originally for music source separation. |
| `sepformer` | SepFormer | Dual-path transformer, intra-chunk and inter-chunk attention. |

### 6. Vision-Inspired (Spectrogram Inpainting)

Re-frames the problem in the time-frequency domain and uses image-style
inpainting.

| Model id (suggested) | Source model in report | Notes |
|---|---|---|
| `vit_spectrogram` | ViT/MAE on spectrogram | Mask GA-dominated regions, train ViT to inpaint. |

### 7. Graph (GNN)

Uses scalp electrode geometry to enforce spatial consistency of the artifact.

| Model id (suggested) | Source model in report | Notes |
|---|---|---|
| `st_gnn` | ST-GNN | Spatiotemporal graph network, electrode-graph adjacency. |

## Already Implemented (Baselines)

Do not respawn agents for these. Use them as comparison baselines.

- `cascaded_dae` (1D-CNN DAE baseline)
- `cascaded_context_dae` (DAE with 7-epoch context window)
- `demo01` (early demo model)

## Cross-Family Hybrids (Optional)

The report explicitly recommends hybrids:

- GNN spatial constraint + Mamba temporal core
- Diffusion artifact prior + GNN spatial prior
- Audio-style TCN + Transformer attention over TR boundaries

Hybrids may be assigned later once at least one model from each pure family is
working and evaluated.

## Selection Heuristic

When the orchestrator is unsure which to assign next:

1. First make sure each pure family from sections 1-7 has at least one
   working agent.
2. Prefer architectures whose primary published code is permissively
   licensed and reproducible.
3. For early agents, prefer models with single-pass inference. Diffusion
   models (D4PM) are valuable but slower to validate.
4. Within a family, pick the model with the clearest published architecture
   description and the highest reported metric in the report.
