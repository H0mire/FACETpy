# Importing the facet module
from facet.facet import facet

# It is advised to add a configuration block here, to keep an overview of the settings used for the analysis.
# Begin Configuration Block
# Path to your BIDS dataset
edf_file_path = "./examples/datasets/NiazyFMRI.edf"
# Event Regex assuming using stim channel
event_regex = r"\b1\b"
# Upsampling factor
upsample_factor = 10
# Assuming you want to use the same path for exporting the data
export_file_path = edf_file_path

# Now some settings for the AAS
window_size = 30
relative_window_position = -0.5
artifact_to_trigger_offset_in_seconds = -0.005
regex_trigger_annotation_filter = r"\bYour Trigger Tag\b"  # Annotations with the description 'Your Trigger Tag' are considered as triggers
unwanted_bad_channels = [
    "EKG",
    "EMG",
    "EOG",
    "ECG",
]  # Channels with these names are considered as bad channels and not considered in the processing
evaluation_measures = [
    "SNR",
    "RMS",
    "RMS2",
    "MEDIAN",
]  # Evaluation measures to be used for the evaluation of the AAS
# End Configuration Block

# Loading the EEG data by creating a facet object and importing the EEG data
f = facet()
f.import_eeg(
    path=edf_file_path,
    fmt="edf",
    upsampling_factor=upsample_factor,
    artifact_to_trigger_offset=artifact_to_trigger_offset_in_seconds,
    bads=unwanted_bad_channels,
)
f.get_eeg().mne_raw.crop(0, 162)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_eeg().copy(), name="Original")
# Preprocessing
f.highpass(1)
f.add_to_evaluate(f.get_eeg().copy(), name="After Highpass")
f.upsample()  # upsampling factor must be specified when importing the EEG data

# Find and load the triggers
f.find_missing_triggers()

# Align the triggers
reference_trigger = 0
f.align_triggers(reference_trigger)
f.align_subsample(reference_trigger)

# Apply the AAS
f.calc_matrix_aas()  # calculates the AAS matrix
f.remove_artifacts()  # calculates the artifacts and removes them from the EEG data
f.add_to_evaluate(f.get_eeg().copy(), name="After AAS")
f.get_correction().apply_PCA(n_components=0.8)  # apply PCA to the EEG data
f.add_to_evaluate(f.get_eeg().copy(), name="After PCA")
# Postprocessing
f.downsample()  # downsampling by upsample factor
f.lowpass(70)  # lowpass filter with 70 Hz
f.add_to_evaluate(f.get_eeg().copy(), name="After Lowpass")
f.apply_ANC()  # apply the ANC to the EEG data. This may take some time. If you want keep track of the progress, you can set the logger level to DEBUG

# Plot the EEG data
f.plot_eeg()

# Evaluation
f.add_to_evaluate(f.get_eeg().copy(), name="Corrected EEG")
results = f.evaluate(plot=True, measures=evaluation_measures)
print(results)

# f.export_eeg(path=export_file_path, fmt="edf")
