# Multi-Source Artifact Library Strategy

## Problem

The current synthetic training dataset is based primarily on one extracted artifact source. This creates a major risk of artifact-specific overfitting. fMRI gradient artifacts are not universal waveforms; they depend on scanner hardware, MRI sequence, slice timing, trigger structure, EEG cap and amplifier characteristics, channel montage, and preprocessing choices.

If training uses only one artifact source, the model may learn a specific artifact image rather than a general correction principle.

## Thesis-Ready Statement

The use of a single artifact source creates a substantial risk of artifact-specific overfitting. Since fMRI-induced gradient artifacts vary across scanners, acquisition sequences, trigger timing, EEG montages, and preprocessing pipelines, training should be based on a multi-source artifact library and extended through domain randomization. Generalization should then be evaluated using leave-one-dataset-out validation, because a random train-validation split within a single artifact source can substantially overestimate transfer performance to unseen EEG-fMRI recordings.

## Recommended Strategy

### 1. Build a Multi-Source Artifact Library

Artifact windows should be extracted from multiple EEG-fMRI recordings and stored in a shared library format.

Each artifact source should include:

- artifact estimate array,
- trigger positions,
- sampling frequency,
- artifact-to-trigger offset,
- channel names,
- original source identifier,
- extraction pipeline or correction method,
- and optional scanner or sequence metadata.

The training dataset builder should sample artifact contexts from all available sources rather than from one fixed Niazy-derived bundle.

### 2. Preserve Source Metadata in the Training Dataset

Each synthetic example should store which artifact source it used. This makes later analysis possible.

Useful fields include:

- `artifact_source_ids`,
- `artifact_source_paths`,
- `artifact_context_indices`,
- `artifact_channel_indices`,
- `artifact_epoch_lengths_samples`,
- `artifact_scales`,
- `artifact_jitter_samples`.

Without these fields, it is difficult to diagnose whether performance depends on a specific artifact source.

### 3. Domain Randomization

If only few real artifact sources are available, additional variation should be introduced during synthetic mixing.

Useful augmentations:

- amplitude scaling,
- non-circular temporal shifts,
- slight time stretching or compression,
- source mixing,
- channel-wise amplitude profiles,
- variable artifact durations,
- additive low-amplitude residual noise,
- randomized trigger phase,
- and variable artifact epoch lengths.

The goal is not to make arbitrary unrealistic signals, but to expose the model to a wider artifact family.

### 4. Artifact Mixture Training

Instead of sampling only one artifact context, a later extension could combine multiple source contexts:

```text
artifact = a * artifact_source_i + b * artifact_source_j + residual
```

This would encourage the model to learn an artifact manifold rather than memorize individual templates.

### 5. Leave-One-Source-Out Evaluation

Once more than one artifact source is available, validation should not only be a random split over all generated windows. A stronger evaluation is:

```text
train: artifact sources A, B, C
test: artifact source D
```

This directly tests whether the model generalizes to unseen artifact morphology.

### 6. Optional Sequence Conditioning

In a later stage, the model may receive explicit source or sequence features:

- trigger delta,
- normalized epoch phase,
- artifact epoch length,
- slice count,
- TR or sequence timing,
- scanner or protocol label if available.

This may help the model adapt its correction to different artifact families. However, this should be added only after the multi-source baseline exists.

## Immediate Implementation Recommendation

The next implementation step should be an `ArtifactLibrary` abstraction in the synthetic dataset builder. It should:

- load one or more `.npz` artifact bundles,
- derive trigger-to-trigger artifact epochs for each source,
- resample each epoch to the model epoch length,
- build multi-epoch contexts per source,
- sample contexts across sources,
- and save source identifiers in the final synthetic dataset.

This does not yet solve the domain gap, but it creates the infrastructure required to address it.
