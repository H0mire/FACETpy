"""
Correction Module

This module contains processors for correcting EEG artifacts.

Author: FACETpy Team
Date: 2025-01-12
"""

from .aas import AASCorrection, AveragedArtifactSubtraction
from .deep_learning import (
    DeepLearningArchitecture,
    DeepLearningChannelGroupingStrategy,
    DeepLearningCorrection,
    DeepLearningDomain,
    DeepLearningDualOutputPolicy,
    DeepLearningExecutionGranularity,
    DeepLearningLatencyProfile,
    DeepLearningModelAdapter,
    DeepLearningModelRegistry,
    DeepLearningModelSpec,
    DeepLearningOutputType,
    DeepLearningPrediction,
    DeepLearningRuntime,
    NumpyInferenceAdapter,
    OnnxInferenceAdapter,
    OnnxTensorLayout,
    PyTorchInferenceAdapter,
    PyTorchTensorLayout,
    TensorFlowInferenceAdapter,
    TensorFlowTensorLayout,
    get_deep_learning_model,
    list_deep_learning_blueprints,
    list_deep_learning_models,
    register_deep_learning_model,
    spec_from_dict,
    spec_to_dict,
)
from .farm import FARMArtifactCorrection, FARMCorrection
from .volume import RemoveVolumeArtifactCorrection, VolumeArtifactCorrection
from .weighted import (
    AvgArtWghtCorrespondingSliceCorrection,
    AvgArtWghtMoosmannCorrection,
    AvgArtWghtSliceTriggerCorrection,
    AvgArtWghtVolumeTriggerCorrection,
    CorrespondingSliceCorrection,
    MoosmannCorrection,
    SliceTriggerCorrection,
    VolumeTriggerCorrection,
)

__all__ = [
    # AAS
    "AASCorrection",
    "AveragedArtifactSubtraction",
    # Deep learning integration
    "DeepLearningArchitecture",
    "DeepLearningChannelGroupingStrategy",
    "DeepLearningRuntime",
    "DeepLearningDomain",
    "DeepLearningOutputType",
    "DeepLearningLatencyProfile",
    "DeepLearningDualOutputPolicy",
    "DeepLearningExecutionGranularity",
    "DeepLearningModelSpec",
    "DeepLearningPrediction",
    "DeepLearningModelAdapter",
    "DeepLearningModelRegistry",
    "NumpyInferenceAdapter",
    "OnnxTensorLayout",
    "OnnxInferenceAdapter",
    "PyTorchTensorLayout",
    "PyTorchInferenceAdapter",
    "TensorFlowTensorLayout",
    "TensorFlowInferenceAdapter",
    "register_deep_learning_model",
    "get_deep_learning_model",
    "list_deep_learning_models",
    "list_deep_learning_blueprints",
    "spec_to_dict",
    "spec_from_dict",
    "DeepLearningCorrection",
    # FARM
    "FARMCorrection",
    "FARMArtifactCorrection",
    # Volume transitions
    "VolumeArtifactCorrection",
    "RemoveVolumeArtifactCorrection",
    # AAS weighting variants
    "CorrespondingSliceCorrection",
    "VolumeTriggerCorrection",
    "SliceTriggerCorrection",
    "MoosmannCorrection",
    "AvgArtWghtCorrespondingSliceCorrection",
    "AvgArtWghtVolumeTriggerCorrection",
    "AvgArtWghtSliceTriggerCorrection",
    "AvgArtWghtMoosmannCorrection",
]

# Import ANC if available
try:
    from .anc import AdaptiveNoiseCancellation, ANCCorrection  # noqa: F401

    __all__.extend(["AdaptiveNoiseCancellation", "ANCCorrection"])
except ImportError:
    pass

# Import PCA if available
try:
    from .pca import PCACorrection  # noqa: F401

    __all__.append("PCACorrection")
except ImportError:
    pass
