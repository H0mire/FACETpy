import os
from facet.facet import facet
from loguru import logger
import sys

from facet.utils.facet_result import FACETResult

bids_path = '/Volumes/JanikSSD/EEG Datasets/openneuro/FMRIWITHMOTION'
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
event_id_description_pairs = {
    "trigger": 1,
    "New Segment/": 2,
    "Response/R128": 3,
    "Stimulus/S  2": 4,
    "Stimulus/S 99": 5,
    "missing_trigger": 6
}
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

# Find triggers
f.find_triggers(regex_trigger_annotation_filter)
f.find_missing_triggers()
f._eeg.loaded_triggers[:-1]

# Now as we have triggers, we can evaluate the data
results_preprocessed = f.evaluate(f.get_eeg(), measures=evaluation_measures, name="preprocessed")

# Now align the triggers
f.align_triggers(0)
f.align_subsample(0)
# Calculate the averaging matrix
f.calc_matrix_aas()
# Calculate the artifacts and remove them
f.remove_artifacts(plot_artifacts=False)
results_after_aas = f.evaluate(f.get_eeg(), measures=evaluation_measures, name="After AAS")
# Now Postprocess the data
f.get_correction().apply_PCA()
f.downsample()
f.lowpass(70)
f.apply_ANC()
results_after_anc = f.evaluate(f.get_eeg(), measures=evaluation_measures, name="After ANC")
f.plot_eeg(title="after postprocessing")
f.export_eeg(
    "example_full_bids",
    fmt="bids",
    subject=subject,
    session=session,
    task=task,
    event_id=event_id_description_pairs,
)
f.plot([results_preprocessed, results_after_aas, results_after_anc], plot_measures=evaluation_measures)

facet_result = FACETResult.from_facet_object(f)
print(facet_result.get_metadata('_tmin'))
facet_result.mne_noise.plot()

input("Press Enter to end the script...")
