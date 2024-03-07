from FACET.Facet import Facet
#It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
#Begin Configuration Block
# Path to your EEG file
file_path = 'NiazyFMRI.edf'
# Event Regex assuming using stim channel
event_regex = r'\b1\b'
# Upsampling factor
upsample_factor = 10
#unwanted channels
unwanted_bad_channels = ['EKG', 'EMG', 'EOG', 'ECG']
#Add relative Trigger Position. Adjust this if the trigger events are not aligned with the artifact occurence
rel_trigger_pos = -0.01
#End Configuration Block

# Loading the EEG data by creating a FACET object and importing the EEG data
f = Facet()
f.import_EEG(file_path, upsampling_factor=upsample_factor, bads=unwanted_bad_channels, rel_trig_pos=rel_trigger_pos)

#f.pre_processing()
f.highpass(1)
f.upsample()
f.find_triggers(event_regex)
f.apply_AAS()
f.remove_artifacts()
#f.post_processing()
f.downsample()
f.lowpass(45)
f.plot_EEG(start=29)
#f.export_EEG('processed_eeg_file.edf')
input("Press Enter to end the script...")
