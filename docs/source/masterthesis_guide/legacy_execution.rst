Phase-0 execution
=================

Original reproduction
---------------------

The original Phase-0 experiment used the earlier FACETpy API. Reproduce it in an
isolated interpreter with the corresponding source revision and dependencies.
Do not import two incompatible packages named ``facet`` into one interpreter.
The recorded checkpoint and available metrics are retained even when exact
original-run provenance is incomplete.

A verified source revision identifies a code snapshot; it does not prove the
original training environment or recover an absent training runner. Consult the
catalog's gaps before claiming a complete numerical replay.

Current pipeline adaptation
---------------------------

The current ``LegacyDLAdapter`` recreates the small network and loads both stages
strictly. It uses the checkpoint's normalization and native dimensions, then
adapts predictions to the current pipeline's epoch boundaries.

This path deliberately changes segmentation and resampling. It demonstrates
current-library use of the legacy model and is labelled as adapted execution.
It is not presented as the original Phase-0 pipeline.

.. literalinclude:: ../../../masterthesis_guide/examples/reproduce_phase0.py
   :language: python

The model was fitted to an AAS-derived target from the same recording. Its
interpretation remains the feasibility result described in
:doc:`../thesis_reference/phase_0_legacy`.
