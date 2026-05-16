# Nested-GAN

Two-stage generator that attacks the fMRI gradient artifact in
**both the time-frequency domain and the time domain**, following the
architectural recipe in *End-to-End EEG Artifact Removal Method via Nested
Generative Adversarial Network* (Biomed. Phys. Eng. Express, 2025;
PMID 41183389).

- **Inner branch.** STFT of the center epoch is processed by a light
  complex-valued Restormer (MDTA + GDFN blocks). It removes the
  TR-locked harmonic content of the gradient artifact in the spectrogram
  representation, then inverse-STFTs back to time.
- **Outer branch.** The full 7-epoch noisy context is fed to a 1D U-Net.
  The center-epoch slot is replaced with the inner branch's output so the
  outer branch refines the waveform and corrects residual phase
  discontinuities across trigger boundaries. The orchestrator's
  DHCT-GAN lesson — "use the full 7-epoch context, not just the center
  epoch" — is honoured here.

Inference uses the same channel-wise context-window pattern as
`cascaded_context_dae`, so the TorchScript checkpoint is independent of
the channel count at inference time. See
[`documentation/research_notes.md`](documentation/research_notes.md) for
the design rationale, paper references, and a list of deliberate scope
reductions compared with the published recipe (notably: no separate
adversarial discriminators — the multi-resolution STFT loss is used
instead).

## Scope

- Input shape: `(batch, 7, 1, 512)`.
- Output shape: `(batch, 1, 512)` predicted center-epoch artifact.
- Per-channel inference — compatible with arbitrary channel counts.
- Trigger metadata required at inference; the processor builds the
  7-epoch context from triggers.
- Output type: `artifact`. The pipeline subtracts it from the raw signal.

## Training

Build the Niazy proof-fit context dataset on the GPU worker:

```bash
uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512
```

Smoke run (single epoch, tiny model):

```bash
uv run facet-train fit --config src/facet/models/nested_gan/training_niazy_proof_fit_smoke.yaml
```

Full run:

```bash
uv run facet-train fit --config src/facet/models/nested_gan/training_niazy_proof_fit.yaml
```

Both runs write a TorchScript checkpoint to
`training_output/<run>/exports/nested_gan.ts`.

## Inference

```python
from facet.models.nested_gan import NestedGANCorrection

context = context | NestedGANCorrection(
    checkpoint_path="training_output/<run>/exports/nested_gan.ts",
    context_epochs=7,
    epoch_samples=512,
)
```

## Direct comparisons

- `cascaded_context_dae` — same dataset, same context, same target.
  Most direct architectural comparison.
- `cascaded_dae` — channel-wise DAE without context.
- `dhct_gan` — known failed result at -7.13 dB on this dataset.
  Nested-GAN must improve over this for the time-frequency
  decomposition hypothesis to hold.
