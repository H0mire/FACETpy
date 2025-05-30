import random
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

# Loading the EEG data by creating a facet object and importing the EEG data
f = facet()
f.import_eeg(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)
f.get_eeg().mne_raw.crop(0, 162)
# f.plot_eeg(start=29)
f.find_triggers(event_regex)

f.pre_processing()

# generate 10 random numbers between 0 and 800
missing_triggers = random.sample(range(0, 800), 10)
# remove triggers from loaded_triggers to simulate missing triggers
for i in missing_triggers:
    f.get_eeg().loaded_triggers.pop(i)

f.get_eeg().loaded_triggers = f.get_eeg().loaded_triggers[15:830]
f.find_missing_triggers()
# remove every second trigger to simulate sub periodic artifacts (calculating Slice triggers)
f.get_eeg().loaded_triggers = f.get_eeg().loaded_triggers[::2]  #
f.get_analysis().derive_parameters()
f.find_missing_triggers()
# print count triggers total
print(f.get_eeg().count_triggers)

f.align_triggers(0)
f.calc_matrix_aas()
f.remove_artifacts(plot_artifacts=False)
f.post_processing()
f.plot_eeg(start=29)
# f.export_eeg('processed_eeg_file.edf')
input("Press Enter to end the script...")
