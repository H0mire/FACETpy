from dataclasses import dataclass
import numpy as np

@dataclass
class TemplateICADetection:
    template_z: np.ndarray
    best_channel: int
    refined_times: list
    accepted_components: list  # indices of accepted ICA components
    component_timecourses: list[np.ndarray]  # timecourses of accepted components
    hrf_regressors: dict  # {'component_idx': {'3s': array, '5s': array, ...}}
    ica: object  # MNE ICA object