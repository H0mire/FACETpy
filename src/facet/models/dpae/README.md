# DPAE (Dual-Pathway Autoencoder)

`dpae` is the FACETpy implementation of the Dual-Pathway Autoencoder from
Xiong et al. 2023 (Frontiers in Neuroscience, "A general dual-pathway network
for EEG denoising"). It is the discriminative-CNN reference baseline of the
deep-learning model family in `docs/research/architecture_catalog.md`.

## Scope

- Input shape: `(batch, 1, samples)`.
- Output shape: `(batch, 1, samples)` predicted artifact (subtracted by
  `DeepLearningCorrection`).
- Default samples: `512` (matches Xiong et al. 2-second segment, also matches
  the Niazy proof-fit context dataset's resampled epoch length).
- Local pathway: small kernels (k=3) with dilation rates 1/2/4/8 to capture
  fine temporal detail.
- Global pathway: large kernels (k=15, 11, 7) with strided pooling to capture
  slow trends and the gradient-pulse envelope.
- Fusion: channel-axis concatenation, batch normalisation, 1x1 conv.
- Decoder: mirror transposed convolutions back to the input length.
- Residual connection: a learned scalar adds a fraction of the input to the
  prediction so the model can start from a near-identity initialisation.
- Activation: SeLU throughout, matching the paper.
- Per-channel inference, so the exported checkpoint is independent of EEG
  channel count.

## Training

Build the Niazy proof-fit dataset (the orchestrator typically passes this as
`--prepare-command` to `fleet.py submit`):

```bash
uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512
```

Then train via the FACETpy training CLI:

```bash
uv run facet-train fit --config src/facet/models/dpae/training_niazy_proof_fit.yaml
```

A smoke configuration (`training_niazy_proof_fit_smoke.yaml`) is provided for
a 1-epoch end-to-end check before requesting the full run.

## Inference

```python
from facet.models.dpae import DualPathwayAutoencoderCorrection

context = context | DualPathwayAutoencoderCorrection(
    checkpoint_path="training_output/<run>/exports/dpae.ts",
    epoch_samples=512,
)
```

The processor needs trigger metadata at inference: each native trigger-to-trigger
epoch is resampled to `epoch_samples`, fed through DPAE, and the predicted
artifact is resampled back to the native length before subtraction.

## Architectural decision

DPAE is by design a single-segment encoder. Unlike `cascaded_context_dae`, it
does not aggregate neighbouring epochs. It is therefore a fair "control"
baseline against more complex sequence/context models (Mamba, Conv-TasNet,
diffusion).
