Legacy correction metrics
=========================

The Phase-0 records preserve the original metric values and their signal scope.
These measurements describe the small legacy model and its AAS-derived target.
Do not combine them with a later holdout ranking without stating the difference
in training data, target and evaluation window.

Refactoring benchmarks form a separate evidence group. They measure API parity,
implementation properties or runtime under their recorded conditions. Neural
quality scores are not a substitute for those measurements.

Use the Phase-0 section of :doc:`../masterthesis_guide/catalog` to find the original
JSON and CSV records. Preserved evidence is distinguished from a newly verified
rerun through each experiment's verification field.
