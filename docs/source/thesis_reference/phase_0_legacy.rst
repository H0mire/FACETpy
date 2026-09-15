Phase 0 — Legacy feasibility reference
======================================

Phase 0 uses a small two-stage fully connected denoising autoencoder developed
for the earlier FACETpy API. Its target is the artifact estimated by AAS from the
same recording used for fitting. Agreement with that target demonstrates fitting
behavior, not an independently validated improvement over AAS.

The original metrics, model state and available run records remain associated
with the Phase-0 experiment. Engineering measurements also belong to the thesis
evidence even when they do not involve a neural model.

Two execution contracts
-----------------------

The original pipeline requires its own compatible FACETpy environment. The current
``LegacyDLAdapter`` instead uses trigger-to-trigger segmentation and resamples
predictions around the model's native epoch length. Loading the same checkpoint
does not make those pipelines identical.

See :doc:`../masterthesis_guide/legacy_execution` for the two entry points and any
remaining prerequisites. :doc:`legacy_metrics` defines the scope of the saved
measurements. Neither adapted inference nor a synthetic smoke test is labelled as
a rerun of the original numerical result.

Checkpoint identity
-------------------

``legacy_delivery`` owns the plotted historical metric run and its separately
trained ``dae_plotted_model.pt``. ``legacy_fc_dae`` owns the earlier diagnostic
checkpoint. Their weights differ. The original Figure-23 generator reads
``legacy_native.npz`` from the earlier run; Figures 21 and 22 use the delivery
records. Keep this source distinction when interpreting the figures.
