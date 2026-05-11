# D4PM Model Card

## Summary

`d4pm` is a single-branch conditional denoising-diffusion model that
predicts the gradient-artifact component of a single-channel EEG
trigger-aligned epoch. It is adapted from the dual-branch D4PM paper
(arXiv 2509.14302). The dual-branch joint posterior sampling of the
original is reduced to a one-branch conditional sampler with a
data-consistency residual term; rationale and trade-offs are recorded in
`documentation/research_notes.md`.

## Intended Role

This model is the FACETpy entry point for the diffusion family. It is
slower at inference than the autoencoder baselines (one forward pass per
sampling step per channel per epoch) but is intended to capture
non-Gaussian, non-mean artifact morphology better than MSE-trained
autoencoders. Use it as the comparison baseline whenever a probabilistic
generative artifact prior is required.

## Input And Output

- Input per inference call: 1 EEG channel, `(samples,)` float32, default
  `samples = 512`.
- Output: `(1, samples)` artifact estimate.
- Subtracted by `DeepLearningCorrection`.

## Compatibility Notes

- Compatible with arbitrary EEG channel counts because inference is
  channel-wise.
- Requires trigger metadata.
- Native artifact lengths may vary; the processor resamples each native
  epoch to the model-domain length and resamples the predicted artifact
  back to the native length before subtraction.
- The checkpoint is coupled to model-domain epoch length (`epoch_samples`).

## Architecture Hyperparameters

`build_model` accepts:

- `epoch_samples`: 512 (must match dataset).
- `num_steps`: 200 (training-time DDPM timesteps).
- `beta_start`, `beta_end`: 1e-4, 0.02 (linear schedule, matches paper).
- `feats`: input projection channels.
- `d_model`, `d_ff`, `n_heads`, `n_layers`: transformer block sizes.
- `embed_dim`: noise-level sinusoidal embedding dim.

The forward pass produces a `(B, 2, T)` tensor where channel 0 is the
predicted noise and channel 1 is the true noise drawn during the
training step. The custom `D4PMEpsilonLoss` returns the L1/L2 distance
between the two.

## Inference Hyperparameters

`D4PMArtifactCorrection`/`D4PMArtifactDiffusionAdapter` accept:

- `sample_steps`: number of reverse-process steps used at inference
  (default 50). Smaller values are faster but lower fidelity.
- `data_consistency_weight`: λ_dc residual reinforcement strength
  (default 0.5). Set to 0.0 to disable and run unconditional sampling.
- `device`: `"cpu"` or `"cuda"`.
- `demean_input` / `remove_prediction_mean`: per-epoch DC removal.

## Checkpoint Format

The canonical inference artifact is a state-dict `.pt` file written by
`PyTorchModelWrapper.save_checkpoint` to
`training_output/<run>/checkpoints/last.pt`. The TorchScript file
exported by the FACETpy CLI is retained only as a smoke-test artifact;
it is not used at inference because diffusion sampling is iterative
Python control flow.

## Evaluation Notes

Use the standard evaluation structure described in
`src/facet/models/evaluation_standard.md`.
