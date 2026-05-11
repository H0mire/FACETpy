# SepFormer

Compact channel-wise SepFormer (dual-path Transformer) for fMRI gradient
artifact removal on the FACETpy seven-epoch trigger-locked context.

## Scope

- Input shape: `(batch, 7, 1, samples)`.
- Output shape: `(batch, 1, samples)`.
- Default samples: `512`.
- Predicts the centre-epoch artifact signal.
- Architecture: 1D-conv encoder → dual-path Transformer blocks
  (intra-chunk + inter-chunk self-attention) → masked decoder → centre
  slice. See `documentation/research_notes.md`.
- Applies per channel, so inference remains compatible with different
  EEG channel counts.
- Requires trigger metadata during inference, because seven trigger-
  defined epochs are reconstructed from the `ProcessingContext`.

## Architectural argument

The dual-path attention design (Subakan et al. 2021) gives the model an
inductive bias that matches the fMRI gradient artifact:
**intra-chunk** attention models morphology within one slice/TR;
**inter-chunk** attention models slow drift across consecutive TRs.

## Training

The Niazy proof-fit context dataset must exist on the worker
(`output/niazy_proof_fit_context_512/`). The training YAML expects the
NPZ at `./output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`.

```bash
uv run facet-train fit --config src/facet/models/sepformer/training_niazy_proof_fit.yaml
```

Smoke run (one epoch, tiny config) for sync / CUDA / export validation:

```bash
uv run facet-train fit --config src/facet/models/sepformer/training_niazy_proof_fit_smoke.yaml
```

## Inference

```python
from facet.models.sepformer import SepFormerArtifactCorrection

context = context | SepFormerArtifactCorrection(
    checkpoint_path="training_output/<run>/exports/sepformer.ts",
    context_epochs=7,
    epoch_samples=512,
)
```
