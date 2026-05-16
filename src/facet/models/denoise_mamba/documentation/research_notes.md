# DenoiseMamba — Research Notes

These notes capture the prior reading and design decisions made before
implementing `denoise_mamba` for FACETpy. The model is the
"Sequence Modeling (State Space / Mamba)" entry from
`docs/research/architecture_catalog.md` and is cited as Section 6.2 of
`docs/research/dl_eeg_gradient_artifacts.pdf`.

## Source papers and references

Primary source paper:

- Liu et al., *DenoiseMamba: An Innovative Approach for EEG Artifact Removal
  Leveraging Mamba and CNN.* IEEE Xplore document **11012652**, IEEE Journal
  of Biomedical and Health Informatics, 2025.
  - PubMed: <https://pubmed.ncbi.nlm.nih.gov/40408214/>
  - IEEE Xplore: <https://ieeexplore.ieee.org/document/11012652/>

Supporting / background material:

- Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces.* arXiv: <https://arxiv.org/abs/2312.00752>
- Dao and Gu, *Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality (SSD)* (Mamba-2). arXiv:
  <https://arxiv.org/abs/2405.21060>
- Gu, Dao, Ermon, Rudra, Ré. *HiPPO: Recurrent Memory with Optimal Polynomial
  Projections.* NeurIPS 2020.
  <https://proceedings.neurips.cc/paper/2020/file/102f0bb6efb3a6128a3c750dd16729be-Paper.pdf>
- Reference Mamba implementations:
  - Official CUDA package `mamba-ssm`: <https://github.com/state-spaces/mamba>
  - Pure-PyTorch reference (`mamba-minimal`):
    <https://github.com/johnma2006/mamba-minimal>

The IEEE full text is paywalled and the abstract on PubMed does not list the
exact ConvSSD hyperparameters. The architectural description below is
therefore reconstructed from the report's Section 6.1–6.2, the abstracts
publicly available, and the underlying Mamba/SSD primary papers.

## Plain-language description

DenoiseMamba is a sequence-to-sequence EEG denoiser. It feeds a noisy EEG
segment through a stack of "ConvSSD" blocks. Each ConvSSD block combines:

1. A short 1D convolution that captures the very steep edges of MR gradient
   pulses on a small temporal neighborhood.
2. A selective state space (SSD / Mamba) layer with input-dependent
   parameters that integrates information across the whole segment with
   linear time complexity.

The convolution gives the model a strong local prior; the SSD layer carries
long-range periodicity (TR boundaries, slice events, slowly drifting
artifact morphology) over the entire segment. The output of the stack is a
1-channel signal of the same length as the input. Following the report
("Application to GA"), the selectivity of the SSM lets the model gate around
massive gradient spikes so they do not corrupt the running hidden state.

For FACETpy we use this architecture as a residual artifact predictor: the
network outputs an estimated gradient artifact for each input window;
`DeepLearningCorrection` subtracts that prediction from the noisy raw
signal.

## Key architectural components

The architecture chosen for the FACETpy implementation closely follows the
ConvSSD recipe but is intentionally lightweight to fit comfortably on one
RTX 5090 (24 GB VRAM) and to leave headroom for the parallel agents.

| Component | Responsibility |
|---|---|
| Input projection | A 1D conv (kernel=7, padding=3) lifting the noisy 1-channel input to `d_model` channels. |
| Stack of ConvSSD blocks | Captures local artifact edges (depthwise 1D conv) and long-range periodicity (selective SSM). Pre-norm residual connections. |
| Output projection | A 1×1 conv that maps `d_model` channels back to 1 channel (the artifact estimate). |
| Final residual | Output is the predicted *artifact*; subtraction is left to `DeepLearningCorrection` (see `evaluation_standard.md`). |

A single ConvSSD block executes:

```
y = x + DepthwiseConv1d(LayerNorm(x))            # local edges
y = y + SelectiveSSM(LayerNorm(y))               # long-range periodicity
```

The selective SSM is implemented in pure PyTorch using the standard Mamba
parameterization (Δ, A, B, C, D), with a sequential discrete-time scan. We
deliberately *do not* depend on the `mamba-ssm` CUDA kernel; see the
"Hardware feasibility" section below for why.

Default block hyperparameters (configurable via the YAML):

| Hyperparameter | Default | Notes |
|---|---|---|
| `d_model` | 64 | Channel width inside the stack. |
| `d_state` | 16 | SSM hidden state expansion (Mamba-1 default). |
| `expand` | 2 | Inner SSM channel expansion factor. |
| `d_conv` | 4 | Depthwise conv width inside each Mamba block. |
| `n_blocks` | 4 | Number of stacked ConvSSD blocks. |
| `dropout` | 0.1 | Applied to the SSM output before the residual add. |

These defaults yield a parameter count well under 1 M, leaving the GPU
mostly idle and giving fast per-epoch wall-clock on the proof-fit dataset.

## Inputs in the original paper vs. our dataset

The DenoiseMamba paper evaluates on the EEGdenoiseNet semi-simulated
benchmark (4514 clean EEG segments mixed with EOG/EMG/ECG artifacts; segment
length 512 samples at 256 Hz per the EEGdenoiseNet contract). The original
paper consumes one channel at a time with input shape `(B, 1, T)` and
predicts the clean (or artifact) signal at the same length.

Our Niazy proof-fit dataset built by
`examples/dataset_building/build_niazy_proof_fit_context_dataset.py` is conceptually the same
single-channel sequence-to-sequence task, but applied to MR gradient
artifacts on real EEG-fMRI recordings:

- `noisy_center` shape: `(N_examples, n_channels, epoch_samples)` where
  `epoch_samples = 512` after resampling each trigger-defined artifact
  epoch.
- `artifact_center` shape: `(N_examples, n_channels, epoch_samples)` — the
  AAS-estimated artifact (the prediction target).
- `clean_center` shape: same — the AAS-corrected EEG (surrogate clean).
- The original `sfreq` of the recording is preserved in the bundle as
  `sfreq` (Niazy bundle is around 5 kHz before resampling). After the
  builder's resampling each trigger epoch is uniformly mapped to 512
  samples, regardless of the native epoch length.

Mapping to DenoiseMamba:

- We train per-channel (channel-wise execution), reusing the
  `ChannelWiseContextArtifactDataset` pattern from `cascaded_context_dae`
  but specialized for `context_epochs=1` (single-epoch denoising). Input is
  reshaped to `(B, 1, 512)` and target is `(B, 1, 512)`.
- This matches DenoiseMamba's published single-channel input format and
  also keeps the inference adapter independent of channel count, the same
  property valued by `cascaded_dae` and `cascaded_context_dae`.
- The `noisy_center` array is used as the network input and
  `artifact_center` is used as the target. The model therefore predicts the
  AAS-estimated artifact (proof-of-fit semantics matching the dataset's
  intent).

## Loss function

The DenoiseMamba paper reports MSE / temporal MSE as the primary loss with
extra spectral terms in some ablations. We use plain MSE for the proof-fit
run and expose `loss.name = "mse" | "l1" | "smooth_l1"` through the same
`build_loss` factory pattern used by `cascaded_context_dae`. This keeps the
training contract identical to the existing baselines so metric comparisons
do not confound architecture and loss.

## Non-obvious training tricks

- **Pre-norm Mamba blocks.** LayerNorm before the depthwise conv and before
  the SSM, with residual connections. This is the standard Mamba/Mamba-2
  recipe and stabilizes training of stacked SSD blocks.
- **Channel mixing via input/output 1D conv** rather than pointwise linear,
  so the model ingests the temporal context immediately. The 1D conv at the
  input has kernel=7 (small enough to not disturb gradient-pulse onsets,
  large enough to seed the SSM with a smoothed local context).
- **Pure-PyTorch selective scan.** Loop over time with discretized
  `A_bar = exp(Δ ⊙ A)`, `B_bar = Δ ⊙ B`. This is tractable because:
  - Sequence length is 512 (small).
  - We use `d_state = 16`, so the per-step inner state matrix is small.
  - No autograd through CUDA kernels keeps the model trivially portable to
    CPU and avoids the `--no-build-isolation` install dance for `mamba-ssm`.
- **AdamW + grad clipping.** Same defaults as `cascaded_context_dae`
  (`lr=1e-3`, `weight_decay=1e-4`, `grad_clip_norm=1.0`).
- **Demean per epoch.** Following `cascaded_context_dae`, both input and
  target are demeaned per channel-epoch. This removes a slow drift that
  would otherwise leak into the artifact estimate and competes with the
  SSM's preferred small-state-amplitude regime.

## Hardware feasibility

Hardware envelope: one RTX 5090, 24 GB VRAM, ~64 GB RAM.

Estimates for the default hyperparameter set on a 512-sample epoch:

- Parameter count: roughly `n_blocks × (depthwise_conv + selective_ssm) +
  in_proj + out_proj`. With `d_model=64`, `d_state=16`, `expand=2`,
  `d_conv=4`, `n_blocks=4`, the order of magnitude is **on the order of
  ~150 k parameters**, well under 1 M.
- Activation memory at batch 64 and sequence length 512 with `d_model=64`:
  on the order of **a few MB** for forward + a few times that for the
  scan's intermediate states. **Far** under 24 GB.
- Wall-clock per epoch: the Niazy proof-fit context dataset
  (`niazy_proof_fit_context_dataset.npz`) currently has ~hundreds of
  context examples × number of EEG channels (~30+) → on the order of
  **a few thousand training items** per epoch. With batch size 64 and a
  pure-PyTorch sequential SSM scan on a 5090, one epoch is expected in
  **well under a minute**, comfortably allowing ~50–100 epochs.
- The pure-PyTorch scan is slower than the CUDA `selective_scan` kernel
  (the CUDA kernel is the headline reason Mamba beats Transformer wall
  clock at very long sequences), but at our sequence length of 512 the
  difference is negligible compared to model wall clock and far below
  available GPU headroom. The portability and reproducibility benefits
  outweigh the speed cost.

If a future agent decides the CUDA kernel is needed (e.g. to scale to 5 kHz
raw segments of 10 000+ samples), they can swap the scan implementation
behind a feature flag without touching the FACETpy adapter or training
contract. We intentionally keep `mamba-ssm` *out* of the project's
dependency list to avoid breaking the parallel agents' `uv sync`.

## Open questions

- The IEEE full text was inaccessible from this environment. Exact published
  hyperparameters (`d_model`, `n_blocks`, dropout schedule, optimizer
  settings) are taken from the `mamba`/`mamba-2` defaults and from the
  values that empirically fit our proof-fit envelope rather than from the
  DenoiseMamba paper's own experimental section. If the paper later proves
  to use materially different hyperparameters, a follow-up agent can adjust
  the YAML; the architecture wiring is the same.
- The DenoiseMamba paper trains on EOG/EMG/ECG artifacts, not MR gradient
  artifacts. The choice of architecture for our task is justified by
  Section 6 of the report (long-range periodicity at high sampling rate),
  not by the original benchmark.
- DenoiseMamba does not formally specify whether the model predicts the
  clean signal or the artifact. We follow FACETpy's existing convention
  (`output_type=ARTIFACT`) so the same `DeepLearningCorrection` subtraction
  semantics apply as for `cascaded_dae` / `cascaded_context_dae`. Switching
  to clean-signal prediction would be a one-line change in
  `DeepLearningModelSpec`.
