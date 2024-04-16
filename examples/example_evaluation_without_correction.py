from facet.facet import facet

# It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
# Begin Configuration Block
# Path to your EEG file
file_path = "C:/Users/janik/Projekte/FACETpy/Datasets/CleanedFMRIBAllen_exp_without_subsample_alignment.edf"
file_path2 = "C:/Users/janik/Projekte/FACETpy/Datasets/Matlab_cleaned_with_ssa.edf"
# Event Regex assuming using stim channel
event_regex = r"\b1\b"
# Upsampling factor
upsample_factor = 10
# unwanted channels
unwanted_bad_channels = ["EKG", "EMG", "EOG", "ECG"]
# Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.004296875
# End Configuration Block

# Loading the EEG data by creating a facet object and importing the EEG data
f = facet()
f.import_eeg(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)
f2 = facet()
f2.import_eeg(
    file_path2,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)

f.get_eeg().mne_raw.crop(0, 162)
f2.get_eeg().mne_raw.crop(0, 162)
f.find_triggers(event_regex)
f2.find_triggers(event_regex)
f.plot_eeg(start=29, title="without SSA")
f.add_to_evaluate(f.get_eeg(), name="without SSA")

f2.lowpass(70)
f2.plot_eeg(start=29, title="with SSA")
f.add_to_evaluate(f2.get_eeg(), name="with SSA")

results = f.evaluate(measures=["SNR", "RMS", "RMS2", "MEDIAN"])
print(results)
# f.export_eeg('processed_eeg_file.edf')
input("Press Enter to end the script...")
