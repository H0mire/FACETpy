"""Shared configuration constants for the Epilepsy single-subject evaluation.


"""

import os

# Kept identical to the literal used by evaluate_subject.py.
project_root = r"D:\Medical Engineering and Analytics\Project\FACETpy"

SFREQ = 500.0
TR = 2.5
TH_RAW = 0.85
HALF_WIN_S = 0.15
MATCH_TOL_S = 0.1
MAT_DIR = os.path.join(project_root, "examples", "datasets", "MAT_Files")
RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "results")
)
