# Demucs (Time-Domain) For fMRI Gradient Artifact Removal

Time-domain Demucs (Défossez et al. 2019, arXiv:1911.13254) adapted from music
source separation to channel-wise EEG gradient-artifact prediction on the Niazy
proof-fit context dataset.

## Scope

- Input shape: `(batch, 1, context_epochs * epoch_samples)`. Default `(1, 3584)`.
- Output shape: `(batch, 1, context_epochs * epoch_samples)` — predicted
  artifact across all 7 epochs of the context.
- The pipeline adapter slices the **center epoch** of the prediction and
  resamples it back to the native trigger-to-trigger length before subtraction.
- Channel-wise inference, so the checkpoint stays compatible with any EEG
  channel count.
- Requires trigger metadata at inference.

## Architecture

- U-Net with `L = 4` symmetric encoder/decoder blocks (the original Demucs uses
  `L = 6` for 11-second 44.1 kHz audio; `L = 4` is the largest depth that does
  not collapse our 3584-sample input below one sample at the bottleneck).
- Encoder block: `Conv1d(K=8, S=4)` + ReLU + `Conv1d(K=1)` + GLU.
- Bottleneck: 2-layer bidirectional LSTM with hidden size = bottleneck channel
  count; output projected back via a linear layer.
- Decoder block: `Conv1d(K=3)` + GLU + `ConvTranspose1d(K=8, S=4)` + ReLU
  (no ReLU on the final block — the artifact waveform may be signed).
- U-Net skip connections sum encoder outputs into the matching decoder layer.
- Weight rescaling at initialization (no batch normalization).
- ~16M parameters at `initial_channels=64`, `lstm_layers=2`.

See `documentation/research_notes.md` for the full architectural derivation
and the paper-to-FACETpy mapping.

## Training

Build the Niazy proof-fit context dataset first:

```bash
uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512
```

Train via the FACETpy CLI:

```bash
uv run facet-train fit --config src/facet/models/demucs/training_niazy_proof_fit.yaml
```

Or via the GPU fleet (the orchestrator runs the dispatcher centrally):

```bash
uv run python tools/gpu_fleet/fleet.py submit \
  --name demucs_niazy_smoke \
  --worktree . \
  --training-config src/facet/models/demucs/training_niazy_proof_fit_smoke.yaml \
  --prepare-command "uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz --target-epoch-samples 512 --context-epochs 7 --output-dir output/niazy_proof_fit_context_512"
```

## Inference

```python
from facet.models.demucs import DemucsCorrection

context = context | DemucsCorrection(
    checkpoint_path="training_output/<run>/exports/demucs.ts",
    context_epochs=7,
    epoch_samples=512,
)
```

## Status

- Author: Müller Janik Michael (FACETpy thesis)
- Reference report section: 7.1.2 (Demucs: Deep Music Separation)
- Companion audio-family model already in the project: `conv_tasnet`.
