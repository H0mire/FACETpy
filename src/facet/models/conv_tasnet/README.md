# Conv-TasNet

Channel-wise time-domain source separator adapted from Luo & Mesgarani
2019 (arXiv:1809.07454). Treats noisy EEG as a mixture of two ordered
sources: clean EEG and gradient artifact. The encoder, dilated TCN
separator, and decoder operate on a single EEG channel of fixed length.

## Scope

- Input shape: `(batch, 1, samples)`.
- Output shape: `(batch, 2, samples)` where source `0` = clean EEG and
  source `1` = gradient artifact.
- Default chunk size: `samples = 512`.
- Inference is per channel, so the same checkpoint works on any EEG
  montage.
- No trigger metadata required at inference time.

## Training

Build the Niazy proof-fit context dataset first (the same one used by
`cascaded_context_dae`):

```bash
uv run python examples/build_niazy_proof_fit_context_dataset.py \
  --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz \
  --target-epoch-samples 512 \
  --context-epochs 7 \
  --output-dir output/niazy_proof_fit_context_512
```

Then train and export the TorchScript checkpoint:

```bash
uv run facet-train fit --config src/facet/models/conv_tasnet/training_niazy_proof_fit.yaml
```

The smoke config trains for one epoch; use it before the full run:

```bash
uv run facet-train fit --config src/facet/models/conv_tasnet/training_niazy_proof_fit_smoke.yaml
```

## Inference

```python
from facet.models.conv_tasnet import ConvTasNetCorrection

result = context | ConvTasNetCorrection(
    checkpoint_path="training_output/<run>/exports/conv_tasnet.ts",
    chunk_size_samples=512,
)
```

## Architectural decision

Conv-TasNet was designed for 4-second 8-kHz speech mixtures, but the
gradient artifact problem on the Niazy proof-fit dataset is a 100 ms
512-sample epoch on a single channel. The implementation here keeps
the original encoder/TCN/decoder structure but tunes the receptive
field and channel widths to the shorter input. See
`documentation/research_notes.md` for the full feasibility analysis
and design choices.

## Related documents

- [`documentation/model_card.md`](documentation/model_card.md)
- [`documentation/research_notes.md`](documentation/research_notes.md)
- [`documentation/evaluations.md`](documentation/evaluations.md)
