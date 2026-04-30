# Edge Effects, Mean-Zero Normalization, and Synthetic-to-Real Domain Gap

## Context

During the first practical training and inference experiments with the seven-epoch context model, several important methodological findings emerged. These findings are relevant for the final architecture and for the methodological discussion in the thesis because they show that technically correct training does not automatically imply physiologically meaningful artifact correction.

The evaluated model setup was:

- input: seven consecutive trigger-defined artifact epochs,
- output: predicted artifact for the center epoch,
- reconstruction: `clean_hat = noisy_center - artifact_hat`,
- training data: synthetic spike EEG plus extracted Niazy fMRI artifact estimates,
- real-data test: Niazy EEG-fMRI example recording.

## Finding 1: Boundary Artifacts Can Dominate Model Output

The initial convolutional baseline used zero-padding in `Conv1d` layers. In the generated plots, this produced a strong outlier at the beginning and end of the predicted artifact epoch. The effect was not merely visual; it was measurable across the dataset.

Observed diagnostic result:

- predicted artifact edge mean absolute amplitude: approximately `2854 uV`,
- predicted artifact center mean absolute amplitude: approximately `416 uV`,
- edge-to-center ratio: approximately `6.86`.

This means that the model produced much larger amplitudes at the boundaries than in the physiologically relevant center region. Since correction is computed as subtraction of the predicted artifact, this boundary error was directly injected into the corrected EEG.

## Interpretation

The boundary artifact likely arose from a combination of:

- zero-padding at convolutional boundaries,
- circular artifact jitter using `np.roll`,
- and fixed-size epoch training where the first and last samples are always special positions.

Zero-padding introduces artificial discontinuities because the model sees zeros outside the signal support, even though real EEG and fMRI artifacts do not suddenly continue as zero. Circular shifting is also problematic because it wraps the end of an artifact epoch to the beginning, which can create non-physiological edge structure.

## Implemented Mitigation

Two changes reduced the edge problem:

- replace zero-padding with reflection padding,
- replace circular jitter with non-circular shifting and edge filling.

After this change, the edge-to-center amplitude ratio dropped from approximately `6.86` to approximately `0.92` in the intermediate retraining run. This showed that the original edge outlier was largely an implementation-induced boundary effect rather than an unavoidable property of the data.

## Finding 2: Dataset-Specific DC Offset Should Not Be Learned

A second problem appeared after the edge artifact was reduced: the model learned a strong DC offset in the artifact prediction. This is undesirable because different EEG datasets, subjects, amplifiers, preprocessing chains, and file formats may carry different baseline offsets.

If the model learns such offsets, it may subtract dataset-specific baseline structure rather than scanner-related artifact morphology. This can degrade generalization and can inject artificial shifts into the corrected EEG.

## Mean-Zero Principle

For artifact morphology learning, the relevant information is mostly the shape and timing of the artifact, not the absolute voltage offset of a recording. Therefore, it is methodologically preferable to remove local mean offsets before model inference and training.

The adopted principle is:

- each input epoch is demeaned per channel before being passed to the model,
- the artifact target is demeaned during training,
- the predicted artifact is demeaned before subtraction.

This does not mean that all EEG preprocessing must globally force the entire recording to zero mean once. Instead, the important point is local and consistent normalization at the model window level.

## Why Local Demeaning Is Preferable to Global Demeaning

Global demeaning of an entire dataset can remove one recording-level offset, but it does not solve local baseline drift or differences between epochs. It also makes the model dependent on how the dataset was cropped or segmented.

Local epoch-wise demeaning is better aligned with the correction task because:

- it removes dataset-specific offset information from the model input,
- it prevents the model from learning trivial baseline shortcuts,
- it keeps training and inference consistent,
- and it makes correction less dependent on the absolute voltage level of a specific dataset.

For artifact prediction, this is especially important because the predicted artifact is subtracted from the EEG. Any DC error in the artifact prediction directly becomes a DC error in the corrected signal.

## Finding 3: Synthetic Performance Improved Strongly, Real Niazy Generalization Only Weakly

After edge-safe augmentation, reflection padding, bias-free final convolution, and mean-zero handling, the ten-epoch training run showed strong supervised performance on the synthetic dataset.

Final synthetic metrics:

- synthetic SNR improvement: `+18.28 dB`,
- synthetic MSE reduction: `98.51 %`,
- artifact correlation: `0.993`,
- artifact MAE: `1.66e-05`.

However, the same model produced only weak improvements on the real Niazy recording.

Final Niazy proxy metrics:

- template RMS reduction: `0.20 %`,
- trigger-locked RMS reduction: `0.20 %`,
- median-template peak-to-peak reduction: `15.78 %`.

## Interpretation of the Synthetic-to-Real Gap

The synthetic evaluation is supervised because both `clean_center` and `artifact_center` are known. The Niazy evaluation is not supervised because there is no true clean EEG reference for the real fMRI recording. Therefore, Niazy metrics are only proxy metrics based on trigger-locked structure.

The strong synthetic improvement combined with weak Niazy improvement suggests a domain gap:

- the model learned the synthetic construction task well,
- but the synthetic distribution does not fully match the real Niazy inference distribution,
- and the extracted AAS-based artifact targets may not represent all real residual artifact structure.

This is an important distinction. A model can perform well on synthetically mixed data while still failing to generalize sufficiently to real EEG-fMRI correction.

The use of a single artifact source creates a substantial risk of artifact-specific overfitting. Since fMRI-induced gradient artifacts vary across scanners, acquisition sequences, trigger timing, EEG montages, and preprocessing pipelines, training should be based on a multi-source artifact library and extended through domain randomization. Generalization should then be evaluated using leave-one-dataset-out validation, because a random train-validation split within a single artifact source can substantially overestimate transfer performance to unseen EEG-fMRI recordings.

## Methodological Consequence

The next research step should not only increase model size or training duration. The core issue is likely the realism and diversity of the training distribution.

Important improvements include:

- more real artifact sources from different recordings and acquisition settings,
- stronger variation of artifact amplitude and timing,
- inclusion of spike-free control examples,
- explicit validation on held-out real EEG-fMRI recordings,
- and comparison against classical correction outputs such as AAS or PCA-based correction.

If real clean ground truth is unavailable, evaluation must combine several indirect criteria:

- trigger-locked residual artifact reduction,
- preservation of non-trigger-locked EEG transients,
- spectral plausibility,
- spike preservation checks,
- and visual inspection of corrected epochs.

## Practical Design Decisions Derived from the Experiment

The following design choices are justified by the experiment:

- use artifact prediction rather than direct clean EEG generation,
- use center-epoch prediction with multi-epoch context,
- avoid zero-padding in temporal convolutional artifact models,
- avoid circular jitter when augmenting artifact windows,
- apply local mean removal consistently during training and inference,
- evaluate edge behavior explicitly,
- and treat synthetic-to-real transfer as a separate validation problem.

## Thesis-Ready Core Statement

The first context-model experiments showed that implementation details such as convolutional padding and temporal jitter can introduce systematic boundary artifacts that dominate the correction result. Replacing zero-padding with reflection padding and avoiding circular artifact shifts reduced these edge effects substantially. A second important finding was that dataset-specific DC offsets should not be learned by the model. Local epoch-wise mean removal of inputs, targets, and predictions made the artifact prediction task better aligned with morphology rather than baseline offset. With these changes, supervised synthetic performance improved strongly, reaching a synthetic SNR improvement of approximately `18.28 dB` and an artifact correlation of approximately `0.993`. However, improvements on the real Niazy recording remained weak, indicating a synthetic-to-real domain gap. This demonstrates that synthetic benchmark success is necessary but not sufficient for validating EEG-fMRI artifact correction; real-data proxy metrics and spike-preservation criteria remain essential.
