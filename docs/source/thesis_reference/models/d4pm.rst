D4PM
====

D4PM learns to remove noise added to a target during training. At inference, a sampler
applies the learned predictor repeatedly, using the observed EEG as a condition. One
training forward pass is not the complete correction procedure.

Family and origin
-----------------

**Architecture:** Conditional diffusion model.

**Origin:** EEG artifact removal.

Shao et al. proposed a dual-branch diffusion model for clean EEG and artifact, with
joint posterior sampling [Shao2025]_. The FACETpy base model retains a simpler
artifact-prediction path.

FACETpy adaptation
------------------

Two feature paths inside a predictor must not be confused with two separate clean and
artifact diffusion branches. The base implementation has the former. The experimental
edition can add the latter with an optional clean branch. The sampler, noise schedule
and target are part of the model contract.

Implementation and input
------------------------

The main package is ``facet.models.masterthesis.d4pm``. The recorded adapter contract is
**single epoch, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base and deployment variants have separate factories. Select their input
packing, output type and normalization together with the checkpoint. A dataset
may store seven epochs even when a single-epoch adapter uses only the centre.

Phase 1 uses the Python training module and its reverse-diffusion sampler.
The original TorchScript stub does not contain that sampler. Phase 2 has no
completed pipeline result for D4PM; preserved training records do not establish
successful deployment.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Variants and lineage
--------------------

* :ref:`experimental/paper_accurate/d4pm <diagram-experimental-paper-accurate-d4pm>`:
  This experimental variant adds continuous noise-level and artifact-class
  conditioning. An optional clean branch supports joint posterior sampling; enable it
  explicitly in the configuration.

* :ref:`masterthesis/d4pm/deployment <diagram-masterthesis-d4pm-deployment>`:
  This deployment variant normalizes the training pair and adds a recovered-waveform
  objective at configured diffusion timesteps. Inference still requires an iterative
  sampler. The retained record contains no completed Phase-2 pipeline result.

* :ref:`masterthesis/d4pm <diagram-masterthesis-d4pm>`:
  A conditional noise predictor learns the artifact distribution for one EEG channel
  and epoch. Artifact inference requires its iterative reverse-diffusion sampler.

Source-paper record
-------------------

Sources: [Shao2025]_. See :doc:`../model_references` for full references.

A source citation explains the design lineage. It does not establish that a
FACETpy checkpoint reproduces the source paper's training or results.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-d4pm` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
