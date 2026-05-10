# DHCT-GAN

Dual-branch hybrid CNN-Transformer generative adversarial denoiser.

Source paper: Cai, M.; Zeng, H.; et al. *DHCT-GAN: Improving EEG Signal Quality
with a Dual-Branch Hybrid CNN-Transformer Network.* MDPI Sensors 25(1) 231 (2025).
<https://www.mdpi.com/1424-8220/25/1/231>

## Scope

- Input shape: `(batch, 1, samples)` — single channel time-domain.
- Output shape (TorchScript artifact head): `(batch, 1, samples)` — predicted
  fMRI gradient artifact.
- Default samples: 512 (matches the Niazy proof-fit dataset).
- Trained per channel; the checkpoint is independent of channel count.
- Requires trigger metadata at inference time to delimit trigger-to-trigger
  artifact epochs.

## Generator architecture

- Stem: two 1D conv layers (k=7, k=3) + BatchNorm + LeakyReLU expand the
  noisy signal to `base_channels` features.
- Encoder: 4 stages, each `CNNBlock` + `LocalGlobalTransformerBlock` + 2x
  average-pool downsample. Channel widths: 16 → 32 → 64 → 128.
- Bottleneck: extra CNNBlock at the lowest resolution.
- Two symmetric decoder branches with skip connections from each encoder
  stage. The clean branch outputs `clean_pred`; the artifact branch outputs
  `artifact_pred`.
- A 3-layer gating network produces per-sample weights that fuse the two
  branches into a final clean estimate `fused_clean = gate * clean_pred +
  (1 - gate) * (input - artifact_pred)`.
- The TorchScript export returns the artifact branch's output so the
  FACETpy `DeepLearningCorrection` subtracts the predicted artifact.

## Discriminator + loss

A PatchGAN 1D discriminator (4 strided layers, channels 16→32→64→128) lives
inside the loss module rather than the generator. The discriminator's own
Adam optimizer (lr `1e-4`) is also private to the loss module. This setup
preserves single-optimizer compatibility with the standard `facet-train`
PyTorch wrapper: the CLI optimizer only updates generator parameters,
while the loss module performs the alternating discriminator step on each
gradient-enabled call.

Generator loss (per batch):

```
L_recon       = L1(artifact_pred, artifact_target)
              + alpha * L1(noisy - artifact_pred, clean_target)
L_adv         = BCEWithLogits(D(artifact_pred), 1)
L_generator   = L_recon + beta * L_adv
```

Discriminator loss (per batch, run with `pred.detach()`):

```
L_disc = 0.5 * BCEWithLogits(D(artifact_target), 1)
       + 0.5 * BCEWithLogits(D(artifact_pred.detach()), 0)
```

Defaults: `alpha = 0.5`, `beta = 0.1`.

## Dataset assumption

Uses `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`.
Each epoch is reshaped from `(examples, channels, samples)` into per-channel
training windows, giving `n_examples * n_channels` = 833 × 30 = **24990
windows** at 512 samples per window.

## Training

```bash
uv run python examples/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512

uv run facet-train fit \
  --config src/facet/models/dhct_gan/training_niazy_proof_fit.yaml
```

The smoke config (`training_niazy_proof_fit_smoke.yaml`) caps to 1 epoch and
1024 windows so a full GPU dispatch finishes in well under a minute.

## Inference

```python
from facet.models.dhct_gan import DHCTGanCorrection

ctx = ctx | DHCTGanCorrection(
    checkpoint_path="training_output/<run>/exports/dhct_gan.ts",
    epoch_samples=512,
    device="cpu",
)
```

## Reductions vs. the published model

See `documentation/research_notes.md`. The two main deviations are
(a) smaller channel widths to fit the dataset and VRAM envelope, and
(b) a single PatchGAN discriminator on the artifact head rather than three
sibling discriminators on `(Y1, Y2, Ypre)`.
