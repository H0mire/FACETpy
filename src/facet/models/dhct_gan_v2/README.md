# DHCT-GAN v2 (multi-epoch context)

Dual-branch hybrid CNN-Transformer generative adversarial denoiser, **v2**:
the generator consumes the full 7-epoch noisy context and predicts the
center-epoch artifact. This is a focused follow-up on `dhct_gan` (v1), which
produced `clean_snr_improvement_db = -7.13` on the Niazy proof-fit dataset
because it only saw the single center epoch — every model with multi-epoch
context on this dataset (e.g. `cascaded_context_dae`, `dpae`) is in
positive-SNR territory while every single-epoch model is in negative territory.

Source paper: Cai, M.; Zeng, H.; et al. *DHCT-GAN: Improving EEG Signal Quality
with a Dual-Branch Hybrid CNN-Transformer Network.* MDPI Sensors 25(1) 231 (2025).
<https://www.mdpi.com/1424-8220/25/1/231>

## Scope

- Input shape: `(batch, context_epochs, samples)` — `context_epochs` defaults
  to 7. The 7 noisy time-domain epochs are stacked along the channel axis
  per scalp channel; the center epoch is at index `context_epochs // 2`.
- Output shape (TorchScript artifact head): `(batch, 1, samples)` — predicted
  fMRI gradient artifact for the **center** epoch only.
- Default samples: 512 (matches the Niazy proof-fit dataset).
- Trained per channel; the checkpoint is independent of channel count.
- Requires trigger metadata at inference time to delimit trigger-to-trigger
  artifact epochs (radius = `context_epochs // 2` so the model can only
  correct epochs whose full context fits within the recording).

## Generator architecture

- Stem: two 1D conv layers (k=7, k=3) + BatchNorm + LeakyReLU mix the
  `context_epochs` input channels into `base_channels` features at every
  sample. **This is the only structural change vs v1.**
- Encoder: 4 stages, each `CNNBlock` + `LocalGlobalTransformerBlock` + 2x
  average-pool downsample. Channel widths: 16 → 32 → 64 → 128.
- Bottleneck: extra CNNBlock at the lowest resolution.
- Two symmetric decoder branches with skip connections from each encoder
  stage. The clean branch outputs `clean_pred`; the artifact branch outputs
  `artifact_pred`.
- A 3-layer gating network produces per-sample weights that fuse the two
  branches into a final clean estimate
  `fused_clean = gate * clean_pred + (1 - gate) * (noisy_center - artifact_pred)`.
- The TorchScript export returns the artifact branch's output so the
  FACETpy `DeepLearningCorrection` subtracts the predicted artifact from the
  center epoch.

## Discriminator + loss

Unchanged from v1: a PatchGAN 1D discriminator (4 strided layers, channels
16→32→64→128) lives inside the loss module rather than the generator. The
discriminator's own Adam optimizer (lr `1e-4`) is also private to the loss
module. This setup preserves single-optimizer compatibility with the standard
`facet-train` PyTorch wrapper: the CLI optimizer only updates generator
parameters, while the loss module performs the alternating discriminator step
on each gradient-enabled call.

Generator loss (per batch):

```
L_recon     = L1(artifact_pred, artifact_target)
            + alpha * L1(noisy_center - artifact_pred, clean_target)
L_adv       = BCEWithLogits(D(artifact_pred), 1)
L_generator = L_recon + beta * L_adv
```

Discriminator loss (per batch, run with `pred.detach()`):

```
L_disc = 0.5 * BCEWithLogits(D(artifact_target), 1)
       + 0.5 * BCEWithLogits(D(artifact_pred.detach()), 0)
```

Defaults: `alpha = 0.5`, `beta = 0.1`.

## Dataset assumption

Uses `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`.
Each (example, channel) yields a training window of shape `(7, 512)` for the
generator and `(3, 512)` for the loss (artifact_target, clean_target,
noisy_center stacked along the channel axis). This gives `n_examples *
n_channels = 833 × 30 = 24990` windows.

The dataset wrapper reads `clean_center` from the NPZ directly so the
consistency-loss term compares against the AAS clean reference rather than
recomputing `noisy_center - artifact_target`.

## Training

```bash
uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512

uv run facet-train fit \
  --config src/facet/models/dhct_gan_v2/training_niazy_proof_fit.yaml
```

The smoke config (`training_niazy_proof_fit_smoke.yaml`) caps to 1 epoch and
1024 windows so a full GPU dispatch finishes in well under a minute.

## Inference

```python
from facet.models.dhct_gan_v2 import DHCTGanV2Correction

ctx = ctx | DHCTGanV2Correction(
    checkpoint_path="training_output/<run>/exports/dhct_gan_v2.ts",
    context_epochs=7,
    epoch_samples=512,
    device="cpu",
)
```

## What changed vs v1

Only the input contract. The encoder, transformer blocks, dual decoders,
gating network, discriminator, loss weights, learning rates, and training
schedule are all identical to v1 — the orchestrator wanted v2 to isolate
**single-epoch input** as the variable. See `documentation/research_notes.md`
for the design rationale and the v1 ranking of hypotheses.
