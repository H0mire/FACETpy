from FACET.Facet import Facet
#It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
#Begin Configuration Block
# Path to your EEG file
file_path = 'src/NiazyFMRI.edf'
# Event Regex assuming using stim channel
event_regex = r'\b1\b'
# Upsampling factor
upsample_factor = 10
#unwanted channels
unwanted_bad_channels = ['EKG', 'EMG', 'EOG', 'ECG']
#Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.005
#End Configuration Block

# Loading the EEG data by creating a FACET object and importing the EEG data
f = Facet()
f.import_EEG(file_path, upsampling_factor=upsample_factor, bads=unwanted_bad_channels, artifact_to_trigger_offset=artifact_to_trigger_offset)
f.get_EEG().mne_raw.crop(0, 162)
f.plot_EEG(start=29)
f.pre_processing()
f.find_triggers(event_regex)
f.align_triggers(0)
f.apply_AAS()
f.remove_artifacts(plot_artifacts=False)
f.post_processing()
f.plot_EEG(start=29)
#f.export_EEG('processed_eeg_file.edf')
input("Press Enter to end the script...")