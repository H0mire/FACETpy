Selected thesis variants by model family
========================================

Choose a model by its architecture, input and output contract. Then choose the
variant and experiment. A family name does not identify a checkpoint or a phase.
The family pages below describe the design and its sources.

.. list-table:: Families and their FACETpy role
   :header-rows: 1
   :widths: 19 26 25 30

   * - Family
     - Architecture
     - Origin
     - Main distinction
   * - :doc:`models/cascaded_dae`
     - Fully connected denoising autoencoder
     - Duffy et al.; EEG-fMRI gradient artifact correction
     - Single-epoch residual cascade; keep the parallel Phase-0 model separate.
   * - :doc:`models/context_dae`
     - Fully connected context autoencoder
     - FACETpy context extension of the Duffy et al. family
     - Seven-epoch context; centre artifact output.
   * - :doc:`models/conv_tasnet`
     - Convolutional source separator
     - Speech separation
     - Ordered clean and artifact sources.
   * - :doc:`models/d4pm`
     - Conditional diffusion model
     - EEG artifact removal
     - Iterative diffusion sampling is required.
   * - :doc:`models/demucs`
     - Waveform encoder-decoder with recurrent processing
     - Music source separation
     - Temporal context from one electrode.
   * - :doc:`models/denoise_mamba`
     - Convolutional and state-space model
     - EEG artifact removal
     - Flat Mamba-1 base versus U-shaped experimental ConvSSD.
   * - :doc:`models/dhct_gan`
     - Dual-branch convolution and attention model
     - EEG artifact removal
     - Base, context V2 and independent-branch strict variants.
   * - :doc:`models/dpae`
     - Dual-pathway convolutional autoencoder
     - EEG denoising
     - Two temporal scales; base artifact versus experimental clean target.
   * - :doc:`models/ic_unet`
     - Multichannel U-Net denoising autoencoder
     - EEG artifact removal using ICA-derived training pairs
     - Frozen ICA in the base; sensor-space experimental default.
   * - :doc:`models/multichannel_demucs`
     - Waveform encoder-decoder with electrode attention
     - FACETpy extension of Demucs
     - Electrode attention; separate diagnostic comparison.
   * - :doc:`models/nested_gan`
     - Spectral generator followed by temporal refinement
     - EEG artifact removal; image-restoration building blocks
     - Generator-only recipes; no full nested adversarial training.
   * - :doc:`models/sepformer`
     - Dual-path attention source separator
     - Speech separation
     - Attention chunks are learned features, not input epochs.
   * - :doc:`models/st_gnn`
     - Spatiotemporal graph convolution network
     - Traffic forecasting and EEG graph modelling
     - Fixed electrode graph and channel order.
   * - :doc:`models/vit_spectrogram`
     - Spectrogram Transformer with masked reconstruction
     - Image recognition and masked image reconstruction
     - Magnitude reconstruction versus deployment complex mask.

Variant names
-------------

* **Base:** the retained family implementation. Read the catalog to identify
  which experiments used it.
* **Deployment:** a separate training and inference contract. Most variants
  normalize inputs and score recovered clean EEG. ViT-Spectrogram also changes
  the reconstruction head; D4PM retains a diffusion-specific objective and sampler.
* **V2:** a family-specific revision. DHCT-GAN V2 belongs to the thesis collection
  because it was evaluated there. The name alone says nothing about accuracy.
* **Strict:** the DHCT-GAN package with independent branches and an adversarial
  training wrapper. Its context and optional electrode bridge remain explicit
  configuration choices.
* **Experimental paper_accurate:** a separate attempt to align selected components
  with source papers. The family page lists changes and remaining adaptations.
* **Legacy:** the migrated Phase-0 implementation. Its evidence remains relevant,
  while its original execution environment is only partly known.

An experiment phase records when and how a model was evaluated. It is separate
from both the architecture family and the variant name. For example, a deployment
package does not by itself establish a completed Phase-2 result.

Comparison and evidence
-----------------------

Read each result together with its dataset, target, preprocessing, checkpoint and
metric protocol. The same architecture can have different results under different
conditions. Those conditions must remain visible.

Code is organized by family. Experiments are organized by phase. The base,
deployment and any evaluated V2 variants remain separate where their contracts
differ. Experimental paper-accurate editions state an implementation goal; their
names do not establish agreement with a source paper.

.. list-table:: Comparison scopes
   :header-rows: 1
   :widths: 12 44 44

   * - Phase
     - Question
     - Limit
   * - 0
     - Can a small model reproduce an AAS-derived artifact target?
     - Same-recording fitting is not independent correction validation.
   * - 1
     - How do the models compare on one fixed proof-fit holdout?
     - Tensor holdout quality does not establish full-pipeline quality.
   * - 2
     - Do deployed models behave correctly inside the correction pipeline?
     - Discontinuous output is invalid even if its residual looks small.
   * - 3
     - How do selected families respond to tuning and revised training data?
     - Joint changes to the dataset and objective do not isolate either effect.

For exact settings and availability, use :doc:`../masterthesis_guide/catalog`.
No model or result is excluded solely because it comes from an earlier phase.
