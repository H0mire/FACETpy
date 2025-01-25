from facet.facet import facet

# It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
# Begin Configuration Block
# Path to your EEG file
file_path = (
    "/home/janik/Documents/Projects/FACETpy/facetpy/examples/datasets/NiazyFMRI.edf"
)

matlab_only_lowpass = "/home/janik/Documents/Projects/FACETpy/facetpy/examples/datasets/matlab_only_lowpass.edf"
matlab_only_aas = "/home/janik/Documents/Projects/FACETpy/facetpy/examples/datasets/matlab_only_aas.edf"
matlab_with_alignment = "/home/janik/Documents/Projects/FACETpy/facetpy/examples/datasets/matlab_with_alignment.edf"
matlab_with_pca = "/home/janik/Documents/Projects/FACETpy/facetpy/examples/datasets/matlab_with_pca.edf"
matlab_with_anc = "/home/janik/Documents/Projects/FACETpy/facetpy/examples/datasets/matlab_with_anc.edf"
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
f.get_eeg().mne_raw.crop(0, 162)
f.get_eeg().mne_raw_orig.crop(0, 162)
f.find_triggers(event_regex)
f.plot_eeg(start=29, title="Original")
f.highpass(1)
f.upsample()
f.align_triggers(0)
f.align_subsample(0)
f.calc_matrix_aas()
f.remove_artifacts()
f.get_correction().apply_PCA(n_components=0.8)
f.downsample()
f.lowpass(70)
f.apply_ANC()
f.plot_eeg(start=29, title="FACETpy")
results_facetpy = f.evaluate(
    eeg=f.get_eeg(),
    name="FACETpy",
    measures=["SNR", "RMS", "RMS2", "MEDIAN"]
)

# Second EEG import
f2 = facet()
f2.import_eeg(
    matlab_with_anc,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)
f2.get_eeg().mne_raw_orig = f.get_eeg().mne_raw_orig
f2.get_eeg().mne_raw.crop(0, 162)
f2.find_triggers(event_regex)
f2.plot_eeg(start=29, title="MATLAB")
results_matlab = f.evaluate(
    f2.get_eeg(), measures=["SNR", "RMS", "RMS2", "MEDIAN"], name="MATLAB"
)

f.plot([results_facetpy, results_matlab], plot_measures=["SNR", "RMS", "RMS2", "MEDIAN"])

# f.export_eeg('processed_eeg_file.edf')
input("Press Enter to end the script...")
