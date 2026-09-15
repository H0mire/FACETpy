"""Paper-accurate edition of the Spatiotemporal Graph Neural Network model.

A more faithful reproduction of Yu/Yin/Zhu (2018) STGCN (arXiv:1709.04875)
for the spatiotemporal block and Wagh/Varatharajah (2020) EEG-GCNN
(arXiv:2011.12107) for the domain-guided electrode graph, while staying
compatible with the FACETpy facet-train / inference contracts and CPU-cheap.

See ``README.md`` and ``documentation/paper_accuracy_review.md`` for the
discrepancy table and EEG-fMRI applicability assessment.
"""

from .processor import (
    PaperAccurateSpatiotemporalGNNAdapter,
    PaperAccurateSpatiotemporalGNNCorrection,
)
from .training import (
    NIAZY_PROOF_FIT_CHANNELS,
    SpatiotemporalGNN,
    build_chebyshev_laplacian,
    build_dataset,
    build_loss,
    build_model,
)

__all__ = [
    "NIAZY_PROOF_FIT_CHANNELS",
    "PaperAccurateSpatiotemporalGNNAdapter",
    "PaperAccurateSpatiotemporalGNNCorrection",
    "SpatiotemporalGNN",
    "build_chebyshev_laplacian",
    "build_dataset",
    "build_loss",
    "build_model",
]
