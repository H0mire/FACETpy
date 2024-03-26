from FACET.Facet import Facet
#It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
#Begin Configuration Block
# Path to your EEG file
file_path_edf_without_anc = 'C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_without_anc.edf'
file_path_edf_with_anc = 'C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_with_anc.edf'
file_path_edf_with_alignment = 'C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_with_alignment.edf'
file_path_edf_with_alignment_subsamplealignment_anc = 'C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_with_alignment_subsamplealignment_anc.edf'
file_path_edf_full = 'C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_full.edf'
file_path_edf_full_fastr = 'C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_full_fastr.edf'

event_regex = r'\b1\b'


#End Configuration Block

# Loading the EEG data by creating a FACET object and importing the EEG data
f = Facet()
f.import_EEG(file_path_edf_without_anc)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_EEG(), name="Without ANC")
f.plot_EEG(start=29, title="Without ANC")

f.import_EEG(file_path_edf_with_anc)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_EEG(), name="With ANC")
f.plot_EEG(start=29, title="With ANC")

f.import_EEG(file_path_edf_with_alignment)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_EEG(), name="With Alignment")

f.import_EEG(file_path_edf_with_alignment_subsamplealignment_anc)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_EEG(), name="With Alignment and ANC")

f.import_EEG(file_path_edf_full)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_EEG(), name="Full")
f.plot_EEG(start=29, title="Full")

f.import_EEG(file_path_edf_full_fastr)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_EEG(), name="Full Fastr")

#Own implementation
file_path = 'NiazyFMRI.edf'
# Event Regex assuming using stim channel
# Upsampling factor
upsample_factor = 10
#unwanted channels
unwanted_bad_channels = ['EKG', 'EMG', 'EOG', 'ECG']
#Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.01
#End Configuration Block

# Loading the EEG data by creating a FACET object and importing the EEG data
f.import_EEG(file_path, upsampling_factor=upsample_factor, bads=unwanted_bad_channels, artifact_to_trigger_offset=artifact_to_trigger_offset)

#f.pre_processing()
f.highpass(1)
f.upsample()
f.get_correction().prepare_ANC()
f.find_triggers(event_regex)
#f.get_correction().align_slices(0)
f.apply_AAS()
f.remove_artifacts(plot_artifacts=False)
#f.post_processing()
f.downsample()
f.lowpass(45)
eeg_without_anc = f.get_EEG().copy()
eeg_without_anc.mne_raw.crop(tmin = 0, tmax = 160)
f.plot_EEG(start=29, title="Own Implementation without ANC", eeg=eeg_without_anc)
f.add_to_evaluate(eeg_without_anc, name="Own Implementation without ANC")
f.get_correction().apply_ANC()
f.get_EEG().mne_raw.crop(tmin = 0, tmax = 160)
f.add_to_evaluate(f.get_EEG(), name="Own Implementation")
f.plot_EEG(start=29, title="Own Implementation")

f.evaluate(plot=True,measures=["SNR", "RMS", "RMS2", "MEDIAN"])

input("Press Enter to end the script...")
