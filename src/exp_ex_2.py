from FACET.Facet import Facet
#It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
#Begin Configuration Block
# Path to your EEG file
file_path = 'src/NiazyFMRI.edf'
file_path2 = 'C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_without_lowpass.edf'
# Event Regex assuming using stim channel
event_regex = r'\b1\b'
# Upsampling factor
upsample_factor = 10
#unwanted channels
unwanted_bad_channels = ['EKG', 'EMG', 'EOG', 'ECG']
#Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.004296875
#End Configuration Block

# Loading the EEG data by creating a FACET object and importing the EEG data
f = Facet()
f.import_EEG(file_path, upsampling_factor=upsample_factor, bads=unwanted_bad_channels, artifact_to_trigger_offset=artifact_to_trigger_offset)
f2 = Facet()
f2.import_EEG(file_path2, upsampling_factor=upsample_factor, bads=unwanted_bad_channels, artifact_to_trigger_offset=artifact_to_trigger_offset)

f.get_EEG().mne_raw.crop(0, 162)
f2.get_EEG().mne_raw.crop(0, 162)
f.pre_processing()
f.find_triggers(event_regex)
f2.find_triggers(event_regex)
f.align_triggers(0)
f.calc_matrix_AAS()
f.remove_artifacts(plot_artifacts=False)
f.post_processing()

f.plot_EEG(start=29, title="Own")
f.add_to_evaluate(f.get_EEG(), name="Own")

f2.lowpass(70)
f2.plot_EEG(start=29, title="Matlab")
f.add_to_evaluate(f2.get_EEG(), name="Matlab")

f.evaluate(measures=["SNR", "RMS", "RMS2", "MEDIAN"])
#f.export_EEG('processed_eeg_file.edf')
input("Press Enter to end the script...")