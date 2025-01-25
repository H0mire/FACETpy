from facet.facet import facet

# It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
# Begin Configuration Block
# Path to your EEG file
file_path = "./examples/datasets/NiazyFMRI.edf"
# Event Regex assuming using stim channel
event_regex = r"\b1\b"
# Upsampling factor
upsample_factor = 10
# unwanted channels
unwanted_bad_channels = ["EKG", "EMG", "EOG", "ECG"]
# Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.005
# End Configuration Block


# start measuring time
import time

start_time = time.time()

f = facet()


f.import_eeg(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
    preload=False,  # This is important to avoid memory issues. Otherwise applying per channel has no effect.
)

f.find_triggers(event_regex)
f.get_eeg().mne_raw.crop(0, 162)
f.find_missing_triggers()
f.align_triggers(0, save=True, ref_channel=2)  #


def apply_per_channel(f):
    f.pre_processing()
    f.align_subsample(0)
    f.calc_matrix_aas()
    f.remove_artifacts(plot_artifacts=True)
    f.post_processing()


f.apply_per_channel(apply_per_channel)

# end measuring time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

f.plot_eeg(start=29)
input("Press Enter to end the script...")
