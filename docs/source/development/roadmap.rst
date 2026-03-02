Roadmap
=======

This document outlines planned development themes and milestones for FACETpy.

Vision
------

FACETpy aims to be the leading Python toolkit for fMRI artifact correction in EEG data,
providing:

- **Best-in-class algorithms** for artifact removal
- **Easy-to-use API** accessible to all skill levels
- **Extensible architecture** for research and production
- **Comprehensive documentation** and examples
- **Active community** of users and contributors

Current Status
--------------

**Version 2.0.0** (released on 2025-10-31) is the current stable major release.

- Processor-based architecture shipped
- Pipeline API and context model stabilized
- Modular processor catalog expanded across I/O, preprocessing, correction, and evaluation
- Documentation and test suite are actively maintained with ongoing updates

Near-Term Milestones
--------------------

**Q2 2026**

- Improve documentation quality gates (strict Sphinx build and runnable examples)
- Expand benchmark coverage for channel-wise execution and batch workflows
- Improve onboarding by consolidating installation and first-run guidance

**Q3 2026**

- Add additional end-to-end examples for BIDS-focused pipelines
- Strengthen error messages and validation hints for common misconfigurations
- Improve visualization and reporting ergonomics for quality metrics

Research And Platform Direction
-------------------------------

- Continue evaluating advanced correction techniques for robust residual artifact handling
- Improve support for large-scale datasets and memory-constrained execution environments
- Maintain tight interoperability with MNE-Python and BIDS tooling
