DHCT-GAN
========

DHCT-GAN learns features for clean EEG and artifact in separate branches. Convolutions
capture local patterns; attention combines information across the signal. Learned gates
combine branch estimates. Discriminators judge predictions during adversarial training
and are not part of the correction signal path.

Family and origin
-----------------

**Architecture:** Dual-branch convolution and attention model.

**Origin:** EEG artifact removal.

Cai, Meng and Huang introduced the dual-branch hybrid CNN-Transformer generative
adversarial network (DHCT-GAN) for EEG denoising [Cai2025]_. FACETpy retains several
adaptations with different branch, context and training contracts.

FACETpy adaptation
------------------

The base and V2 models share an encoder between their output branches. V2 adds
neighbouring epochs as input features. The strict implementation uses independent
branches and returns clean, noise and fused outputs for the adversarial training
wrapper. Its optional electrode-attention bridge is a FACETpy extension. The words
strict and paper_accurate identify packages, not verified equivalence of every
configuration to the paper.

Implementation and input
------------------------

The main package is ``facet.models.masterthesis.dhct_gan``. The recorded adapter contract is
**single epoch for the base implementation; seven epochs for V2, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base and deployment variants have separate factories. Select their input
packing, output type and normalization together with the checkpoint. A dataset
may store seven epochs even when a single-epoch adapter uses only the centre.

The V2 implementation is under ``dhct_gan.v2``. It remains part of the thesis
collection because the thesis evaluates it. The two incomplete Phase-3 DHCT
trials do not constitute a completed 27-configuration tuning study.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Variants and lineage
--------------------

* :ref:`experimental/paper_accurate/dhct_gan <diagram-experimental-paper-accurate-dhct-gan>`:
  This experimental variant adds separate fusion gates, least-squares adversarial
  losses and discriminator feature matching. Its shared encoder and reduced EEG
  configuration remain FACETpy adaptations.

* :ref:`experimental/paper_accurate/dhct_gan_v2 <diagram-experimental-paper-accurate-dhct-gan-v2>`:
  This experimental context variant uses separate fusion gates and adversarial feature
  matching. It retains a shared encoder and returns artifact as noisy centre EEG minus
  the fused clean estimate.

* :ref:`masterthesis/dhct_gan/deployment <diagram-masterthesis-dhct-gan-deployment>`:
  A shared encoder and two decoders estimate clean EEG and artifact from a
  single-channel epoch. The exported forward path returns the artifact branch. The
  deployment wrapper normalizes input, restores output units and can remove the
  predicted epoch mean. Its objective scores recovered clean EEG.

* :ref:`masterthesis/dhct_gan <diagram-masterthesis-dhct-gan>`:
  A shared encoder and two decoders estimate clean EEG and artifact from a
  single-channel epoch. The exported forward path returns the artifact branch.

* :ref:`masterthesis/dhct_gan/strict <diagram-masterthesis-dhct-gan-strict>`:
  Independent CNN/Transformer branches return clean, noise and fused outputs for
  adversarial training. An optional electrode-attention bridge and configurable
  decoder extend the source design; identify both in any reported result.

* :ref:`masterthesis/dhct_gan/v2/deployment <diagram-masterthesis-dhct-gan-v2-deployment>`:
  The V2 core reads neighbouring epochs as input features and predicts the centre
  artifact. The deployment wrapper normalizes input, restores output units and can
  remove the predicted epoch mean. Its objective scores recovered clean EEG.

* :ref:`masterthesis/dhct_gan/v2 <diagram-masterthesis-dhct-gan-v2>`:
  This thesis V2 variant stacks seven epochs as features for one electrode. A shared
  encoder and dual decoders predict the centre artifact. It differs from both the
  single-epoch base and the experimental paper_accurate V2 edition.

Source-paper record
-------------------

Sources: [Cai2025]_. See :doc:`../model_references` for full references.

A source citation explains the design lineage. It does not establish that a
FACETpy checkpoint reproduces the source paper's training or results.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-dhct-gan` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
