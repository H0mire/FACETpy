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
upsampling_factor = 4
artifact_to_trigger_offset_in_seconds = -0.038
relative_window_position = -0.5
event_id_description_pairs = {"trigger": 1}
regex_trigger_annotation_filter = r"\bResponse\b"  # Annotations with the description 'trigger' are considered as triggers
unwanted_bad_channels = [
    "EKG",
    "EMG",
    "EOG",
    "ECG",
]  # Channels with these names are considered as bad channels and not considered in the processing
evaluation_measures = ["SNR", "RMS", "RMS2", "MEDIAN"]

# start measuring time
import time

start_time = time.time()

f = facet()


f.import_eeg(
    path=bids_path,
    fmt="bids",
    upsampling_factor=upsampling_factor,
    artifact_to_trigger_offset=artifact_to_trigger_offset_in_seconds,
    bads=["EKG", "EMG", "EOG", "ECG"],
    subject="xp101",
    session=None,
    task="eegfmriNF",
    preload=False,  # This is important to avoid memory issues. Otherwise applying per channel has no effect.
)

f.find_triggers(regex_trigger_annotation_filter)
f.find_missing_triggers()  # ensure that all triggers are found
f.align_triggers(0)


def apply_per_channel(f):
    f.pre_processing()
    f.align_subsample(0)
    f.calc_matrix_aas()
    f.remove_artifacts(plot_artifacts=False)
    f.post_processing()


f.apply_per_channel(apply_per_channel)

# end measuring time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

f.plot_eeg(start=29)
input("Press Enter to end the script...")
