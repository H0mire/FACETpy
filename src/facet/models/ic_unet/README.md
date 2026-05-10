# IC-U-Net

Multichannel 1-D U-Net with frozen ICA preprocessing, adapted from
Chuang et al. 2022 (PubMed 36031182, arXiv 2111.10026, reference repo
`roseDwayane/AIEEG`).

## Scope

- Input shape: `(batch, 30, 7*512=3584)` — the 7-epoch noisy context flattened
  along time.
- Output shape: `(batch, 30, 512)` — predicted center-epoch artifact.
- Treats every channel as a putative independent component. A frozen 30×30
  unmixing matrix `W` (initialised from `sklearn.decomposition.FastICA` fit on
  the training data) maps the input into IC space; the U-Net denoises in IC
  space; the pseudoinverse `W⁺` maps the cleaned signal back into channel
  space; the center 512-sample window of (noisy − clean) is returned as the
  predicted artifact.
- Multichannel (not channel-wise) — the checkpoint is tied to the 30-channel
  Niazy proof-fit montage; retrain for other channel counts.
- Requires trigger metadata at inference time (same constraint as
  `cascaded_context_dae`).
- Crucially uses the **full 7-epoch context**, not a single epoch — addresses
  the DHCT-GAN lesson where single-epoch input was the suspected failure
  cause.

## Training

The Niazy proof-fit context dataset must exist at
`output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`. On
the GPU fleet, build it via the `--prepare-command` passed at submit time.

```bash
uv run facet-train fit --config src/facet/models/ic_unet/training_niazy_proof_fit.yaml
```

## Inference

```python
from facet.models.ic_unet import IcUnetCorrection

context = context | IcUnetCorrection(
    checkpoint_path="training_output/<run>/exports/ic_unet.ts",
    context_epochs=7,
    epoch_samples=512,
)
```

## Architectural decisions

- **Multichannel input, multichannel output.** The published IC-U-Net treats
  all 30 ICs as input channels of a single U-Net — not one U-Net per IC.
- **U-Net core ported from `cumbersome_model2.UNet1`.** Channel ladder
  `30 → 64 → 128 → 256 → 512` and back; kernel sizes `7, 7, 5, 3, 3, 3, 3, 1`.
- **`LeakyReLU(0.1)` instead of `Sigmoid`.** EEG amplitudes are unbounded;
  a sigmoid output would squash the prediction range. Recorded in
  `documentation/research_notes.md`.
- **Frozen ICA, learnable U-Net.** Keeps the published `ICA + U-Net` framing
  while making the trainable part stable. The `W` matrix is baked into the
  TorchScript checkpoint as a `register_buffer`.
- **`artifact_center` head.** The model outputs the center-epoch artifact so
  the loss applies directly to the same target used by `cascaded_context_dae`
  for a like-for-like comparison.
- **Ensemble loss.** Default for the full run; the smoke run uses plain MSE
  to isolate any infrastructure issues.

## References

- `documentation/research_notes.md` — paper, math, architectural decisions.
- `documentation/model_card.md` — input/output contract, limitations.
- `documentation/evaluations.md` — pointers to evaluation runs (populated
  after full training completes).
