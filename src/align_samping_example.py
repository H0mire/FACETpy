from FACET.Facet import Facet

# It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
# Begin Configuration Block
# Path to your EEG file
file_path = "NiazyFMRI.edf"
# Event Regex assuming using stim channel
event_regex = r"\b1\b"
# Upsampling factor
upsample_factor = 10
# unwanted channels
unwanted_bad_channels = ["EKG", "EMG", "EOG", "ECG"]
# Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.01
# End Configuration Block

# Loading the EEG data by creating a FACET object and importing the EEG data
f = Facet()
f.import_EEG(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)
eeg_without_alignment = f.get_EEG()

# f.pre_processing()
f.highpass(1)
f.upsample()
f.find_triggers(event_regex)
f.calc_matrix_AAS()
f.remove_artifacts(plot_artifacts=False)
# f.post_processing()
f.downsample()
f.lowpass(45)

# Import seconds time but this time align slices before correction
f.import_EEG(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)
f.highpass(1)
f.upsample()
f.find_triggers(event_regex)
f.get_correction().align_triggers(0)
f.calc_matrix_AAS()
f.remove_artifacts(plot_artifacts=True)
f.downsample()
f.lowpass(45)
eeg_with_alignment = f.get_EEG()

# now evaluate the difference
f.add_to_evaluate(eeg_without_alignment, name="Without Alignment")
f.add_to_evaluate(eeg_with_alignment, name="With Alignment")
f.evaluate(plot=True, measures=["SNR", "RMS", "RMS2", "MEDIAN"])

# f.export_EEG('processed_eeg_file.edf')
input("Press Enter to end the script...")
