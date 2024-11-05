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
eeg_without_alignment = f.get_eeg()

# f.pre_processing()
f.highpass(1)
f.upsample()
f.find_triggers(event_regex)
f.calc_matrix_aas()
f.remove_artifacts(plot_artifacts=False)
# f.post_processing()
f.downsample()
f.lowpass(45)

# Import seconds time but this time align slices before correction
f.import_eeg(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)
f.get_eeg().mne_raw.crop(0, 162)
f.highpass(1)
f.upsample()
f.find_triggers(event_regex)
f.align_triggers(0)
f.calc_matrix_aas()
f.remove_artifacts(plot_artifacts=False)
f.downsample()
f.lowpass(45)
eeg_with_alignment = f.get_eeg()

# Import seconds time but this time align slices before correction
f.import_eeg(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)
f.get_eeg().ssa_hp_frequency = 300
f.get_eeg().mne_raw.crop(0, 162)
f.highpass(1)
f.upsample()
f.find_triggers(event_regex)
f.align_triggers(0)
f.get_correction().align_subsample(0)
f.calc_matrix_aas()
f.remove_artifacts(plot_artifacts=False)
f.downsample()
f.lowpass(45)
eeg_with_subalignment = f.get_eeg()

# now evaluate the difference
results_without_alignment = f.evaluate(eeg=eeg_without_alignment, name="Without Alignment", plot=False, measures=["SNR", "RMS", "RMS2", "MEDIAN"])
results_with_alignment = f.evaluate(eeg=eeg_with_alignment, name="With Alignment", plot=False, measures=["SNR", "RMS", "RMS2", "MEDIAN"])
results_with_subalignment = f.evaluate(eeg=eeg_with_subalignment, name="With Subalignment", plot=False, measures=["SNR", "RMS", "RMS2", "MEDIAN"])

f.plot([results_without_alignment, results_with_alignment, results_with_subalignment], plot_measures=["SNR", "RMS", "RMS2", "MEDIAN"])

# f.export_eeg('processed_eeg_file.edf')
input("Press Enter to end the script...")
