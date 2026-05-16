# DenoiseMamba (ConvSSD)

DenoiseMamba is a sequence-modeling artifact denoiser inspired by the IEEE
paper *DenoiseMamba: An Innovative Approach for EEG Artifact Removal
Leveraging Mamba and CNN* and by Section 6.2 of
`docs/research/dl_eeg_gradient_artifacts.pdf`. It stacks ConvSSD blocks that
combine a local 1D depthwise convolution with a Mamba-1 style selective state
space (SSD) layer.

The selective scan is implemented in pure PyTorch so the model trains on the
GPU fleet without depending on the `mamba-ssm` CUDA kernel and stays
testable on CPU.

## Scope

- Input shape: `(batch, 1, samples)`.
- Output shape: `(batch, 1, samples)` predicted gradient artifact.
- Default `samples=512` (matches the Niazy proof-fit context bundle).
- Channel-wise inference, so the checkpoint is independent of EEG channel
  count.
- Trigger metadata is **not** required at inference time — chunks are slid
  over the raw signal in steps of `chunk_size_samples`.

See [`documentation/research_notes.md`](documentation/research_notes.md) for
the architectural reasoning, hyperparameter justification, and references.

## Build the proof-fit dataset

```bash
uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512
```

The DenoiseMamba dataset factory consumes the same NPZ bundle but uses only
the per-epoch `noisy_center` / `artifact_center` arrays (no temporal
context). Building it once for the full `cascaded_context_dae` recipe
therefore also produces what DenoiseMamba needs.

## Train

Smoke run (one epoch, tiny model, ~1 minute on RTX 5090):

```bash
uv run facet-train fit \
  --config src/facet/models/denoise_mamba/training_niazy_proof_fit_smoke.yaml
```

Full run:

```bash
uv run facet-train fit \
  --config src/facet/models/denoise_mamba/training_niazy_proof_fit.yaml
```

## Inference

```python
from facet.models.denoise_mamba import DenoiseMambaCorrection

context = context | DenoiseMambaCorrection(
    checkpoint_path="training_output/<run>/exports/denoise_mamba.ts",
    chunk_size_samples=512,
    device="cuda",
)
```

## Architectural decision

We deliberately stick to single-epoch denoising rather than the 7-epoch
context layout used by `cascaded_context_dae`. The published DenoiseMamba
recipe is itself a single-channel sequence-to-sequence model and the
selective SSM already covers the long-range periodicity in a 512-sample
segment. This also keeps the architecture comparison clean against
`cascaded_dae`, which is the natural single-epoch baseline.

The pure-PyTorch selective scan trades raw wall-clock against portability:
it is slower than the `mamba-ssm` CUDA kernel on very long sequences but is
fast enough at 512 samples and does not require the `--no-build-isolation`
install dance for `mamba-ssm`. Future agents can swap the scan
implementation behind the same module boundary if they need to scale to
multi-second 5 kHz segments.

## Related documents

- [`documentation/research_notes.md`](documentation/research_notes.md) — full
  research reading and architectural reasoning.
- [`documentation/model_card.md`](documentation/model_card.md) — formal model
  card.
- [`documentation/evaluations.md`](documentation/evaluations.md) — index of
  evaluation runs.
- [`../evaluation_standard.md`](../evaluation_standard.md) — required
  evaluation outputs.
