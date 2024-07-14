import os
from facet.facet import facet
from loguru import logger
import sys

bids_path = "F:\EEG Datasets\openneuro\FMRIWITHMOTION"
export_bids_path = bids_path

# configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.add("facet.log", level="DEBUG")

window_size = 30
upsampling_factor = 1
artifact_to_trigger_offset_in_seconds = -0.038
relative_window_position = -0.5
moosmann_motion_threshold = 0.8
event_id_description_pairs = {"trigger": 1}
# Annotations with the description 'trigger' are considered as triggers
regex_trigger_annotation_filter = r"\bResponse\b"
unwanted_bad_channels = [
    "EKG",
    "EMG",
    "EOG",
    "ECG",
]  # Channels with these names are considered as bad channels and not considered in the processing
evaluation_measures = ["SNR", "RMS", "RMS2", "MEDIAN"]

# define which EEG Data of the BIDS Dataset to import
subject = "xp101"
session = None
task = "eegfmriNF"

f = facet()

f.import_eeg(
    path=bids_path,
    fmt="bids",
    upsampling_factor=upsampling_factor,
    artifact_to_trigger_offset=artifact_to_trigger_offset_in_seconds,
    bads=unwanted_bad_channels,
    subject=subject,
    session=session,
    task=task,
)
f.plot_eeg(title="after import")

# Do some preprocessing
f.highpass(1)
f.upsample()
f.plot_eeg(title="after preprocessing")
f.add_to_evaluate(f.get_eeg(), "preprocessed")

# Find triggers
f.find_triggers(regex_trigger_annotation_filter)
f.find_missing_triggers()
# Now align the triggers
f.align_triggers(0)
f.align_subsample(0)
# Calculate the averaging matrix
f.calc_matrix_aas()
# Calculate the artifacts and remove them
f.remove_artifacts(plot_artifacts=False)
f.add_to_evaluate(f.get_eeg(), "After AAS")
# Now Postprocess the data
f.get_correction().apply_PCA()
f.downsample()
f.lowpass(70)
f.apply_ANC()
f.add_to_evaluate(f.get_eeg(), "After ANC")
f.plot_eeg(title="after postprocessing")
f.export_eeg(
    "example_full_bids",
    fmt="bids",
    subject=subject,
    session=session,
    task=task,
    event_id=event_id_description_pairs,
)
f.evaluate(plot=True, measures=evaluation_measures)

input("Press Enter to end the script...")
