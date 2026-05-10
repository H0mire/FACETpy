# DHCT-GAN v2 Research Notes

## Source

- Cai, M.; Zeng, H.; et al. *DHCT-GAN: Improving EEG Signal Quality with a Dual-Branch
  Hybrid CNN-Transformer Network.* MDPI Sensors **25**(1), 231 (2025).
  DOI: 10.3390/s25010231. <https://www.mdpi.com/1424-8220/25/1/231>
- Related GAN-style EEG denoising background: *GAN-Guided Parallel CNN-Transformer
  Network for EEG Denoising* (PubMed 37220036), and the pix2pix conditional GAN
  setup (Isola et al., 2017).

## What v2 changes vs v1

V1 lived under `worktrees/model-dhct_gan` and produced
`clean_snr_improvement_db = -7.13` on the Niazy proof-fit dataset (worse than
no correction). Its hand-off ranked three hypotheses; the orchestrator
selected the second one — **single-epoch input is not enough context for
gradient artifacts** — for v2 to address in isolation:

> Gradient artifacts are strongly periodic across TR boundaries; a model that
> sees only the center epoch cannot exploit that periodicity. `cascaded_dae`
> (single-epoch DAE) is also negative on this benchmark, while every model
> with multi-epoch context is positive.

V2 therefore makes the **single** change of feeding the full 7-epoch noisy
context (the `noisy_context` array from
`output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`,
shape `(833, 7, 30, 512)`) into the generator, with the artifact head still
predicting the **center** epoch only. Everything else — encoder, transformer
stack, dual decoders, gate, discriminator, optimizer betas, learning rates,
adversarial weight, consistency weight, schedule — is identical to v1 so that
the change can be attributed cleanly.

The other two v1 hypotheses (adversarial term destabilizing the generator;
mean-mismatch in the consistency loss) are *not* addressed in v2. If v2 also
underperforms after the input-contract change, the orchestrator will treat
that as confirmation that the GAN training recipe is the actual blocker on
this dataset (not the input contract), and the model card will be updated
accordingly.

## Input-handling design choice

The v1 hand-off described two ways to consume the 7-epoch context:

- **(a) Stack the 7 context epochs as 7 additional input channels per scalp
  channel and produce the center-epoch artifact prediction.**
- (b) Treat the context dimension as a sequence and use attention across it
  inside the transformer branch.

**V2 implements (a).** Reasons:

1. The first conv layer in the stem already mixes input channels at every
   sample. Stacking the 7 epochs there gives the local CNN kernels direct
   cross-epoch covariance access — the dataset is already trigger-aligned by
   the builder, so corresponding samples across epochs are at corresponding
   indices. This is exactly the signal a periodic gradient artifact
   provides.
2. Approach (b) would require adding a second attention axis (`B, T, K, C`
   tensors) to every `LocalGlobalTransformerBlock`. That is more invasive,
   adds parameters, and complicates TorchScript tracing. The orchestrator
   asked for a narrow-scope follow-up.
3. `cascaded_context_dae` — the strongest autoencoder so far — effectively
   does (a) by flattening `(7, 1, 512)` into a single fully-connected input.
   The 1D conv stem in DHCT-GAN is the natural CNN analogue of that
   pattern: instead of flattening, the kernels learn per-sample
   cross-epoch features.

The only structural diff in code is:

| Layer | v1 | v2 |
|---|---|---|
| `stem[0]` | `Conv1d(in=1, out=base_channels, k=7)` | `Conv1d(in=7, out=base_channels, k=7)` |
| `forward` precondition | `x.shape[1] == 1` | `x.shape[1] == context_epochs` |
| `fused_clean` reference | `gate * clean_pred + (1-gate) * (x - artifact_pred)` | `gate * clean_pred + (1-gate) * (noisy_center - artifact_pred)` where `noisy_center = x[:, center_idx:center_idx+1, :]` |

The discriminator still operates on `(B, 1, T)` artifact tensors. The loss
contract is unchanged: `pred = (B, 1, T)`, `target = (B, 3, T)` packing
`[artifact, clean, noisy_center]`.

## Plain-language description (carried over from v1)

DHCT-GAN is a conditional generative adversarial network for EEG denoising.
The generator is a U-shaped hybrid CNN-Transformer encoder followed by two
parallel decoders: one decoder predicts the clean EEG, the other predicts
the artifact. A learned gating network produces per-sample mixing weights
that fuse the two branches. V2 keeps that structure but conditions the
encoder on the full 7-epoch context so the transformer's global attention
and the CNN's local kernels can both leverage TR-period periodicity.

## Mapping to the FACETpy Niazy proof-fit dataset

| Array | Shape | Use |
|---|---|---|
| `noisy_context` | `(833, 7, 30, 512)` | Generator input (per-channel slice gives `(7, 512)`) |
| `clean_center` | `(833, 30, 512)` | Consistency loss target |
| `artifact_center` | `(833, 30, 512)` | Reconstruction loss target (also the model's output target) |
| `noisy_center` | `(833, 30, 512)` | Used by the consistency-loss arithmetic; also packed into the loss `target` tensor |
| `sfreq` | `4096.0` | Native; each epoch is resampled to 512 by the builder |

Per-channel flattening: `833 * 30 = 24990` training windows; 80% / 20% split
matches v1.

## Loss function

Identical to v1:

```
clean_pre  = noisy_center - artifact_pred   # implicit, used inside loss
L_recon    = L1(artifact_pred, artifact_target)
           + alpha * L1(noisy_center - artifact_pred, clean_target)
L_adv      = BCEWithLogits(D(artifact_pred), 1)
L_total    = L_recon + beta * L_adv
```

with `alpha = 0.5`, `beta = 0.1`, `disc_lr = 1e-4` and the alternating
discriminator step happening inside the loss module on each
gradient-enabled forward pass.

One small correction relative to v1: v1's dataset wrapper computed the
consistency target with a *different* mean than the artifact target (the
v1 hand-off flagged this as hypothesis #3). V2's dataset wrapper still
applies the same v1-style per-window demeaning to keep training schedule
identical, **but** the consistency loss inside `DHCTGanV2Loss` now uses
`noisy_center` directly (read from the dataset's `clean_center` field via
the wrapper) instead of subtracting the artifact estimate from a possibly
differently-demeaned `noisy`. This removes one minor source of gradient
noise without changing the schedule. If v2 still underperforms, both
hypothesis #1 (adversarial destabilization) and the broader recipe
question can be tested in a v3.

## Reductions vs. the published model (unchanged from v1)

| Aspect | Published | Adopted |
|---|---|---|
| Segment length | 1024 samples | 512 samples |
| Encoder depth | 5 stages | 4 stages |
| Channel widths | 64–1024 | 16–128 |
| Transformer | local-only + global | local + global, 4 heads, depth 1 |
| Discriminators | 3 (clean, noise, fused) | 1 (PatchGAN on artifact) |
| Input channels | 1 | **7 (context_epochs)** ← only v2 change |

## Parameter and memory budget

- Generator: ~2.7 M parameters (an additional ~700 input-channel weights for
  the stem are negligible).
- PatchGAN discriminator: ~0.3 M parameters.
- Batch 64 × 7 channels × 512 samples per item: model-side activations still
  comfortably under 1 GB on RTX 5090.

## Wall-clock per epoch

Per-batch forward+backward cost is dominated by transformer FFN and conv
operations on `base_channels` features, both unchanged from v1. Expected
wall-clock is similar to v1's 12 s/epoch. The full 80-epoch budget should
finish well under 15 minutes on RTX 5090. The smoke job (1 epoch, 1024
windows) finishes in well under 30 s including dataset load.

## Open questions

1. If v2 metrics still come out worse than `cascaded_context_dae`'s
   `+3.16 dB` (and far below DPAE's `+7.48 dB`), the residual gap is most
   likely attributable to the adversarial training recipe (hypothesis #1 in
   the v1 hand-off). The orchestrator should treat that as the conclusion
   and not chase further hyperparameter sweeps from this agent.
2. The `clean_center` reference is drawn from the AAS template; v2 inherits
   v1's limitation that "clean" really means "what AAS thinks is clean".
   Generalization to non-AAS targets is out of scope for the proof-fit
   evaluation.
